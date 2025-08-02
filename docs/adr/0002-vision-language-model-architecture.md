# ADR-0002: Vision-Language Model Architecture

Date: 2025-01-01
Status: Accepted

## Context

The project needs to select appropriate pre-trained vision-language models for fine-tuning on ultra-low-resource languages. The models must support multilingual text generation while maintaining strong visual understanding capabilities.

## Decision

We will support multiple model architectures with adapters:

1. **Primary: mBLIP-MT0-XL** - Multilingual BLIP variant with mT0 text decoder
2. **Secondary: LLaVA-1.5** - With multilingual LLM backbone replacements
3. **Experimental: BLIP-2** - With multilingual T5 variants

The system will use a model adapter pattern to enable easy switching between architectures.

## Consequences

**Positive:**
- Leverages existing multilingual capabilities
- Flexible architecture supporting multiple models
- Strong baseline performance from pre-trained models
- Community support and active development

**Negative:**
- Model size limitations for deployment
- Potential bias inherited from pre-training
- Limited fine-tuning data for target languages
- Higher computational requirements

## Alternatives Considered

1. **Train from scratch** - Rejected due to insufficient computational resources
2. **English-only models** - Rejected due to poor cross-lingual transfer
3. **Text-only multilingual models** - Rejected due to lack of visual understanding
4. **Proprietary models** - Rejected due to licensing and privacy concerns