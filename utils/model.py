from transformers import ViTForImageClassification


def build_model(num_classes, model_name='google/vit-base-patch16-224-in21k', use_checkpointing=True):
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    
    if use_checkpointing:
        model.gradient_checkpointing_enable()
    
    return model