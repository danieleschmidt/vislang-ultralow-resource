"""Command line interface for VisLang-UltraLow-Resource."""

import click
import logging
from pathlib import Path
from typing import List, Optional
import json

from .scraper import HumanitarianScraper
from .dataset import DatasetBuilder
from .trainer import VisionLanguageTrainer
from .database import DatabaseManager, get_database_manager, get_session
from .database.repositories import DocumentRepository, ImageRepository, DatasetRepository

logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--log-file', type=str, help='Log file path')
def cli(verbose: bool, log_file: Optional[str]):
    """VisLang-UltraLow-Resource: Vision-Language models for humanitarian applications."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure logging
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


@cli.command()
@click.option('--sources', '-s', multiple=True, required=True,
              help='Source organizations (unhcr, who, unicef, wfp, ocha, undp)')
@click.option('--languages', '-l', multiple=True, required=True,
              help='Target languages (e.g., sw, am, ha, ar, fr)')
@click.option('--date-range', nargs=2, type=str,
              help='Date range (start end) in YYYY-MM-DD format')
@click.option('--max-docs', type=int, default=50,
              help='Maximum documents per source')
@click.option('--cache-dir', type=Path, default=Path('./cache'),
              help='Cache directory for downloaded files')
@click.option('--save-db', is_flag=True, help='Save results to database')
@click.option('--output', '-o', type=Path, help='Output JSON file')
def scrape(
    sources: List[str],
    languages: List[str], 
    date_range: Optional[tuple],
    max_docs: int,
    cache_dir: Path,
    save_db: bool,
    output: Optional[Path]
):
    """Scrape humanitarian organization reports."""
    click.echo(f"🔍 Scraping from sources: {', '.join(sources)}")
    click.echo(f"📝 Languages: {', '.join(languages)}")
    
    # Initialize scraper
    scraper = HumanitarianScraper(
        sources=list(sources),
        languages=list(languages),
        date_range=date_range
    )
    
    # Scrape documents
    with click.progressbar(length=len(sources), label='Scraping sources') as bar:
        documents = scraper.scrape(cache_dir=cache_dir, max_docs=max_docs)
        bar.update(len(sources))
    
    click.echo(f"✅ Scraped {len(documents)} documents")
    
    # Save to database if requested
    if save_db:
        with get_session() as session:
            doc_repo = DocumentRepository(session)
            saved_docs = []
            
            for doc in documents:
                try:
                    # Check if document already exists
                    existing = doc_repo.get_by_url(doc['url'])
                    if not existing:
                        db_doc = doc_repo.create(
                            url=doc['url'],
                            title=doc['title'],
                            source=doc['source'],
                            language=doc['language'],
                            content=doc['content'],
                            content_type=doc['content_type'],
                            word_count=doc['word_count'],
                            metadata={'images': doc.get('images', [])}
                        )
                        saved_docs.append(db_doc)
                except Exception as e:
                    logger.error(f"Failed to save document {doc['url']}: {e}")
            
            click.echo(f"💾 Saved {len(saved_docs)} new documents to database")
    
    # Save to file if requested
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        click.echo(f"📄 Saved results to {output}")
    
    # Show statistics
    stats = scraper.get_statistics(documents)
    click.echo("\n📊 Scraping Statistics:")
    click.echo(f"  Total documents: {stats['total_documents']}")
    click.echo(f"  Sources: {stats['sources']}")
    click.echo(f"  Languages: {stats['languages']}")
    click.echo(f"  Total words: {stats['total_words']:,}")
    click.echo(f"  Total images: {stats['total_images']}")


@cli.command()
@click.option('--input', '-i', type=Path, help='Input JSON file with scraped documents')
@click.option('--from-db', is_flag=True, help='Load documents from database')
@click.option('--languages', '-l', multiple=True, required=True,
              help='Target languages')
@click.option('--source-lang', default='en', help='Source language')
@click.option('--min-quality', type=float, default=0.8, help='Minimum quality threshold')
@click.option('--output-dir', type=Path, default=Path('./datasets'),
              help='Output directory')
@click.option('--format', 'output_format', type=click.Choice(['hf_dataset', 'coco', 'custom']),
              default='hf_dataset', help='Output format')
@click.option('--include-infographics/--no-infographics', default=True)
@click.option('--include-maps/--no-maps', default=True)
@click.option('--include-charts/--no-charts', default=True)
@click.option('--name', default='vislang-dataset', help='Dataset name')
def build_dataset(
    input: Optional[Path],
    from_db: bool,
    languages: List[str],
    source_lang: str,
    min_quality: float,
    output_dir: Path,
    output_format: str,
    include_infographics: bool,
    include_maps: bool,
    include_charts: bool,
    name: str
):
    """Build vision-language dataset from scraped documents."""
    click.echo(f"🏗️  Building dataset for languages: {', '.join(languages)}")
    
    # Load documents
    documents = []
    if from_db:
        with get_session() as session:
            doc_repo = DocumentRepository(session)
            db_docs = doc_repo.get_all(limit=10000)  # TODO: add filtering
            
            for db_doc in db_docs:
                doc_dict = {
                    'url': db_doc.url,
                    'title': db_doc.title,
                    'source': db_doc.source,
                    'language': db_doc.language,
                    'content': db_doc.content,
                    'content_type': db_doc.content_type,
                    'word_count': db_doc.word_count,
                    'images': db_doc.metadata.get('images', []) if db_doc.metadata else []
                }
                documents.append(doc_dict)
        
        click.echo(f"📥 Loaded {len(documents)} documents from database")
    
    elif input and input.exists():
        with open(input, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        click.echo(f"📥 Loaded {len(documents)} documents from {input}")
    
    else:
        click.echo("❌ No input source specified. Use --input or --from-db")
        return
    
    # Initialize dataset builder
    builder = DatasetBuilder(
        target_languages=list(languages),
        source_language=source_lang,
        min_quality_score=min_quality,
        output_dir=output_dir
    )
    
    # Build dataset
    with click.progressbar(length=len(documents), label='Processing documents') as bar:
        dataset = builder.build(
            documents=documents,
            include_infographics=include_infographics,
            include_maps=include_maps,
            include_charts=include_charts,
            output_format=output_format
        )
        bar.update(len(documents))
    
    # Save dataset
    output_path = builder.save_dataset(dataset, name, output_format)
    click.echo(f"✅ Dataset saved to {output_path}")
    
    # Show statistics
    stats = builder.get_dataset_statistics(dataset)
    click.echo("\n📊 Dataset Statistics:")
    click.echo(f"  Total items: {stats['total_items']}")
    click.echo(f"  Splits: {stats['splits']}")
    click.echo(f"  Languages: {dict(stats['languages'])}")
    click.echo(f"  Image types: {dict(stats['image_types'])}")
    click.echo(f"  Average quality: {stats['avg_quality_score']:.3f}")


@cli.command()
@click.option('--dataset-path', type=Path, required=True, help='Path to dataset')
@click.option('--model', default='facebook/mblip-mt0-xl', help='Base model')
@click.option('--languages', '-l', multiple=True, required=True, help='Target languages')
@click.option('--output-dir', type=Path, default=Path('./models/finetuned'), help='Output directory')
@click.option('--epochs', type=int, default=10, help='Training epochs')
@click.option('--batch-size', type=int, default=8, help='Batch size')
@click.option('--learning-rate', type=float, default=5e-5, help='Learning rate')
@click.option('--use-wandb', is_flag=True, help='Use Weights & Biases logging')
def train(
    dataset_path: Path,
    model: str,
    languages: List[str],
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    use_wandb: bool
):
    """Train vision-language model."""
    click.echo(f"🚀 Training model {model} for languages: {', '.join(languages)}")
    
    # Load dataset
    from datasets import load_from_disk
    try:
        dataset = load_from_disk(str(dataset_path))
        click.echo(f"📥 Loaded dataset from {dataset_path}")
    except Exception as e:
        click.echo(f"❌ Failed to load dataset: {e}")
        return
    
    # Initialize model and processor
    from transformers import AutoProcessor, AutoModelForVision2Seq
    try:
        processor = AutoProcessor.from_pretrained(model)
        model_obj = AutoModelForVision2Seq.from_pretrained(model)
        click.echo(f"🤖 Loaded model {model}")
    except Exception as e:
        click.echo(f"❌ Failed to load model: {e}")
        return
    
    # Initialize trainer
    trainer = VisionLanguageTrainer(
        model=model_obj,
        processor=processor,
        languages=list(languages),
        use_wandb=use_wandb
    )
    
    # Start training
    click.echo("🏋️  Starting training...")
    results = trainer.train(
        train_dataset=dataset['train'],
        eval_dataset=dataset.get('validation', dataset.get('test')),
        output_dir=str(output_dir),
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    click.echo("✅ Training completed!")
    click.echo(f"📊 Final metrics:")
    click.echo(f"  Train Loss: {results['final_train_loss']:.4f}")
    click.echo(f"  Eval Loss: {results['final_eval_loss']:.4f}")
    click.echo(f"  BLEU Score: {results['final_eval_bleu']:.4f}")
    click.echo(f"  Model saved to: {results['output_dir']}")


@cli.command()
def init_db():
    """Initialize database schema."""
    click.echo("🗄️  Initializing database...")
    
    try:
        db_manager = get_database_manager()
        db_manager.create_tables()
        click.echo("✅ Database initialized successfully")
        
        # Check health
        health = db_manager.health_check()
        click.echo(f"📊 Health check: {health}")
        
    except Exception as e:
        click.echo(f"❌ Database initialization failed: {e}")


@cli.command()
def db_stats():
    """Show database statistics."""
    click.echo("📊 Database Statistics:")
    
    try:
        with get_session() as session:
            # Document statistics
            doc_repo = DocumentRepository(session)
            doc_stats = doc_repo.get_statistics()
            
            click.echo(f"\n📄 Documents:")
            click.echo(f"  Total: {doc_stats['total_documents']}")
            click.echo(f"  By source: {doc_stats['by_source']}")
            click.echo(f"  By language: {doc_stats['by_language']}")
            click.echo(f"  Quality distribution: {doc_stats['quality_distribution']}")
            
            # Image statistics
            img_repo = ImageRepository(session)
            img_stats = img_repo.get_statistics()
            
            click.echo(f"\n🖼️  Images:")
            click.echo(f"  Total: {img_stats['total_images']}")
            click.echo(f"  By type: {img_stats['by_type']}")
            click.echo(f"  Processing status: {img_stats['processing_status']}")
            click.echo(f"  Average confidence: {img_stats['avg_confidence']:.3f}")
            
            # Dataset statistics
            dataset_repo = DatasetRepository(session)
            dataset_stats = dataset_repo.get_statistics()
            
            click.echo(f"\n📊 Dataset Items:")
            click.echo(f"  Total: {dataset_stats['total_items']}")
            click.echo(f"  By split: {dataset_stats['by_split']}")
            click.echo(f"  By language: {dict(dataset_stats['by_language'])}")
            click.echo(f"  Average quality: {dataset_stats['avg_quality']:.3f}")
            
    except Exception as e:
        click.echo(f"❌ Failed to get statistics: {e}")


@cli.command()
@click.option('--port', type=int, default=8000, help='Server port')
@click.option('--host', default='0.0.0.0', help='Server host')
def serve(port: int, host: str):
    """Start API server for inference."""
    click.echo(f"🌐 Starting API server on {host}:{port}")
    
    try:
        import uvicorn
        from .api import app
        
        uvicorn.run(app, host=host, port=port)
    except ImportError:
        click.echo("❌ FastAPI/Uvicorn not installed. Install with: pip install fastapi uvicorn")
    except Exception as e:
        click.echo(f"❌ Failed to start server: {e}")


if __name__ == '__main__':
    cli()