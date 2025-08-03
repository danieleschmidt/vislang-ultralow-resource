"""Vision-language model training functionality."""

from typing import Dict, List, Optional, Any, Union, Callable
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    AutoProcessor, AutoModelForVision2Seq, 
    TrainingArguments, Trainer,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from datasets import Dataset
import numpy as np
from PIL import Image
import json
from pathlib import Path
import wandb
from sklearn.metrics import accuracy_score, f1_score
import evaluate
from tqdm.auto import tqdm
import gc
from accelerate import Accelerator

logger = logging.getLogger(__name__)


class VisionLanguageDataset(TorchDataset):
    """Custom dataset for vision-language training."""
    
    def __init__(self, hf_dataset: Dataset, processor: Any, max_length: int = 512):
        self.dataset = hf_dataset
        self.processor = processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Create a dummy image (in real implementation, load actual image)
        image = Image.new('RGB', (224, 224), color='white')
        
        # Format instruction and response
        instruction = item['instruction']
        response = item['response']
        
        # Process with the model processor
        inputs = self.processor(
            images=image,
            text=instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        
        # Process target text
        targets = self.processor.tokenizer(
            response,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        
        # Prepare labels (replace pad tokens with -100)
        labels = targets['input_ids'].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }


class VisionLanguageTrainer:
    """Trainer for multilingual vision-language models."""
    
    def __init__(
        self,
        model: Any,
        processor: Any,
        languages: List[str],
        instruction_style: str = "natural",
        use_wandb: bool = False,
        wandb_project: str = "vislang-ultralow"
    ):
        """Initialize vision-language trainer.
        
        Args:
            model: Pre-trained vision-language model
            processor: Model processor/tokenizer
            languages: List of target languages
            instruction_style: Instruction generation style
            use_wandb: Enable Weights & Biases logging
            wandb_project: W&B project name
        """
        self.model = model
        self.processor = processor
        self.languages = languages
        self.instruction_style = instruction_style
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        
        # Initialize accelerator for distributed training
        self.accelerator = Accelerator()
        
        # Move model to device
        self.device = self.accelerator.device
        self.model = self.model.to(self.device)
        
        # Initialize metrics
        self.bleu_metric = evaluate.load("bleu")
        self.rouge_metric = evaluate.load("rouge")
        self.bertscore_metric = evaluate.load("bertscore")
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'eval_loss': [],
            'eval_bleu': [],
            'eval_rouge': [],
            'eval_bertscore': [],
            'learning_rates': []
        }
        
        logger.info(f"Initialized trainer for languages: {languages}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        output_dir: str = "./models/vislang-finetuned",
        num_epochs: int = 10,
        learning_rate: float = 5e-5,
        warmup_steps: int = 1000,
        gradient_checkpointing: bool = True,
        batch_size: int = 8,
        eval_batch_size: int = 16,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 100,
        early_stopping_patience: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the vision-language model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir: Directory to save model
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            gradient_checkpointing: Enable gradient checkpointing
            batch_size: Training batch size
            eval_batch_size: Evaluation batch size
            weight_decay: Weight decay factor
            max_grad_norm: Maximum gradient norm
            save_steps: Steps between model saves
            eval_steps: Steps between evaluations
            logging_steps: Steps between logging
            early_stopping_patience: Patience for early stopping
            **kwargs: Additional training arguments
            
        Returns:
            Training results and metrics
        """
        logger.info("Starting vision-language model training")
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Evaluation samples: {len(eval_dataset)}")
        
        # Initialize wandb if enabled
        if self.use_wandb:
            wandb.init(
                project=self.wandb_project,
                config={
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'num_epochs': num_epochs,
                    'warmup_steps': warmup_steps,
                    'weight_decay': weight_decay,
                    'languages': self.languages
                }
            )
        
        # Create datasets
        train_torch_dataset = VisionLanguageDataset(train_dataset, self.processor)
        eval_torch_dataset = VisionLanguageDataset(eval_dataset, self.processor)
        
        # Create data loaders
        train_dataloader = DataLoader(
            train_torch_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        eval_dataloader = DataLoader(
            eval_torch_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Prepare optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        total_steps = len(train_dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Prepare with accelerator
        self.model, optimizer, train_dataloader, eval_dataloader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, eval_dataloader, scheduler
        )
        
        # Enable gradient checkpointing
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Training loop
        best_eval_loss = float('inf')
        patience_counter = 0
        global_step = 0
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            self.model.train()
            total_train_loss = 0
            train_steps = 0
            
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")):
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                train_steps += 1
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                
                global_step += 1
                
                # Logging
                if global_step % logging_steps == 0:
                    avg_loss = total_train_loss / train_steps
                    current_lr = scheduler.get_last_lr()[0]
                    
                    logger.info(f"Step {global_step}, Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
                    
                    if self.use_wandb:
                        wandb.log({
                            'train_loss': avg_loss,
                            'learning_rate': current_lr,
                            'epoch': epoch + 1,
                            'step': global_step
                        })
                
                # Evaluation
                if global_step % eval_steps == 0:
                    eval_results = self.evaluate(eval_dataloader)
                    
                    logger.info(f"Evaluation at step {global_step}: {eval_results}")
                    
                    if self.use_wandb:
                        wandb.log(eval_results)
                    
                    # Early stopping check
                    if eval_results['eval_loss'] < best_eval_loss:
                        best_eval_loss = eval_results['eval_loss']
                        patience_counter = 0
                        
                        # Save best model
                        self.save_model(f"{output_dir}/best", save_processor=True)
                    else:
                        patience_counter += 1
                        
                        if patience_counter >= early_stopping_patience:
                            logger.info(f"Early stopping triggered at step {global_step}")
                            break
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    self.save_model(f"{output_dir}/checkpoint-{global_step}")
            
            # End of epoch evaluation
            avg_train_loss = total_train_loss / len(train_dataloader)
            eval_results = self.evaluate(eval_dataloader)
            
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['eval_loss'].append(eval_results['eval_loss'])
            self.training_history['eval_bleu'].append(eval_results.get('eval_bleu', 0))
            self.training_history['eval_rouge'].append(eval_results.get('eval_rouge_l', 0))
            self.training_history['learning_rates'].append(scheduler.get_last_lr()[0])
            
            logger.info(f"Epoch {epoch + 1} completed:")
            logger.info(f"  Train Loss: {avg_train_loss:.4f}")
            logger.info(f"  Eval Loss: {eval_results['eval_loss']:.4f}")
            logger.info(f"  Eval BLEU: {eval_results.get('eval_bleu', 0):.4f}")
            
            if patience_counter >= early_stopping_patience:
                break
            
            # Cleanup
            gc.collect()
            torch.cuda.empty_cache()
        
        # Save final model
        self.save_model(f"{output_dir}/final", save_processor=True)
        
        # Final evaluation
        final_eval_results = self.evaluate(eval_dataloader)
        
        if self.use_wandb:
            wandb.log(final_eval_results)
            wandb.finish()
        
        results = {
            'final_train_loss': self.training_history['train_loss'][-1],
            'final_eval_loss': final_eval_results['eval_loss'],
            'final_eval_bleu': final_eval_results.get('eval_bleu', 0),
            'final_eval_rouge': final_eval_results.get('eval_rouge_l', 0),
            'training_history': self.training_history,
            'total_steps': global_step,
            'best_eval_loss': best_eval_loss,
            'output_dir': output_dir
        }
        
        logger.info(f"Training completed with results: {results}")
        return results
    
    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on dataset.
        
        Args:
            eval_dataloader: Evaluation data loader
            
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating model")
        
        self.model.eval()
        total_loss = 0
        total_steps = 0
        
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # Forward pass
                outputs = self.model(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                total_steps += 1
                
                # Generate predictions for metrics
                generated = self.model.generate(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=256,
                    num_beams=4,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
                
                # Decode predictions and references
                batch_predictions = self.processor.tokenizer.batch_decode(
                    generated, skip_special_tokens=True
                )
                
                # Get reference texts (labels)
                labels = batch['labels'].clone()
                labels[labels == -100] = self.processor.tokenizer.pad_token_id
                batch_references = self.processor.tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )
                
                predictions.extend(batch_predictions)
                references.extend(batch_references)
        
        avg_loss = total_loss / total_steps
        
        # Calculate metrics
        metrics = {'eval_loss': avg_loss}
        
        try:
            # BLEU score
            bleu_result = self.bleu_metric.compute(
                predictions=predictions,
                references=[[ref] for ref in references]
            )
            metrics['eval_bleu'] = bleu_result['bleu']
            
            # ROUGE score
            rouge_result = self.rouge_metric.compute(
                predictions=predictions,
                references=references
            )
            metrics['eval_rouge_l'] = rouge_result['rougeL']
            
            # BERTScore (sample subset due to computational cost)
            if len(predictions) > 100:
                sample_indices = np.random.choice(len(predictions), 100, replace=False)
                sample_predictions = [predictions[i] for i in sample_indices]
                sample_references = [references[i] for i in sample_indices]
            else:
                sample_predictions = predictions
                sample_references = references
            
            bertscore_result = self.bertscore_metric.compute(
                predictions=sample_predictions,
                references=sample_references,
                lang="en"  # Default to English, could be made configurable
            )
            metrics['eval_bertscore'] = np.mean(bertscore_result['f1'])
            
        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def save_model(self, output_dir: str, save_processor: bool = True) -> None:
        """Save model and processor.
        
        Args:
            output_dir: Directory to save model
            save_processor: Whether to save processor
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save model
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(output_dir)
        
        # Save processor
        if save_processor:
            self.processor.save_pretrained(output_dir)
        
        # Save training history
        with open(Path(output_dir) / "training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Model saved to {output_dir}")
    
    def load_model(self, model_dir: str) -> None:
        """Load model from directory.
        
        Args:
            model_dir: Directory containing saved model
        """
        self.model = AutoModelForVision2Seq.from_pretrained(model_dir)
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = self.model.to(self.device)
        
        # Load training history if available
        history_file = Path(model_dir) / "training_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                self.training_history = json.load(f)
        
        logger.info(f"Model loaded from {model_dir}")
    
    def generate_response(
        self, 
        image: Image.Image, 
        instruction: str, 
        max_length: int = 256,
        num_beams: int = 4,
        temperature: float = 1.0
    ) -> str:
        """Generate response for given image and instruction.
        
        Args:
            image: Input image
            instruction: Text instruction
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        self.model.eval()
        
        # Process inputs
        inputs = self.processor(
            images=image,
            text=instruction,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )
        
        # Decode response
        response = self.processor.tokenizer.decode(
            generated[0], skip_special_tokens=True
        )
        
        return response
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'target_languages': self.languages,
            'instruction_style': self.instruction_style,
            'device': str(self.device),
            'training_history': self.training_history
        }
    
    def create_inference_pipeline(self) -> Callable:
        """Create inference pipeline for deployment."""
        def inference_fn(image: Image.Image, instruction: str) -> str:
            return self.generate_response(image, instruction)
        
        return inference_fn