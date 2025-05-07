import torch
def predict_allele(multi_model, peptide_sequence, thresholds=None):
    """
    Predicts which allele (if any) the peptide is likely to bind to
    
    Args:
        multi_model: The trained MultiAlleleEnsemble
        peptide_sequence: A string of 9 amino acids
        thresholds: Dictionary mapping allele to threshold values
        
    Returns:
        Tuple of (predicted_allele, confidence_score) or (None, 0) if not detected
    """
    if thresholds is None:
        # Default thresholds from your hyperparameters
        thresholds = {
            'A0101': 0.5,
            'A0201': 0.45,
            'A0203': 0.5,
            'A0207': 0.52,
            'A0301': 0.4,
            'A2402': 0.55
        }
    
    # Convert peptide to tensor
    from dataset import peptide_to_indices
    peptide_tensor = torch.tensor([peptide_to_indices(peptide_sequence)], dtype=torch.long)
    
    # Get predictions
    with torch.no_grad():
        logits = multi_model(peptide_tensor)  # Shape: (1, 6)
        probs = torch.sigmoid(logits)         # Convert to probabilities
    
    # Create a dictionary of predictions
    alleles = multi_model.allele_order
    predictions = {allele: float(prob) for allele, prob in zip(alleles, probs[0])}
    
    # Check if any prediction exceeds its threshold
    detected_alleles = []
    for allele, prob in predictions.items():
        if prob >= thresholds.get(allele, 0.5):
            detected_alleles.append((allele, prob))
    
    # Sort by confidence (highest first)
    detected_alleles.sort(key=lambda x: x[1], reverse=True)
    
    # Print all predictions for debugging
    print(f"Peptide: {peptide_sequence}")
    for allele, prob in sorted(predictions.items()):
        threshold = thresholds.get(allele, 0.5)
        status = "✓" if prob >= threshold else "✗"
        print(f"  {allele}: {prob:.4f} (threshold: {threshold}) {status}")
    
    # Return the most confident prediction, or None if none detected
    if detected_alleles:
        return detected_alleles[0]
    else:
        return (None, 0.0)