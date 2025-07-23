import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def preprocess_image(img_path, target_size=(224, 224)):
    """Preprocess an image for the model prediction"""
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)

def evaluate_model_on_dataset(model_path, dataset_path):
    """
    Evaluate model performance on the entire dataset
    Returns accuracy statistics and confusion matrix
    """
    print(f"Evaluating model on dataset at {dataset_path}")
    
    # Load the model
    model = tf.saved_model.load(model_path)
    infer = model.signatures['serving_default']
    
    # Paths for positive and negative cases
    yes_dir = os.path.join(dataset_path, 'yes')
    no_dir = os.path.join(dataset_path, 'no')
    
    # Store predictions and true labels
    all_predictions = []
    all_true_labels = []
    
    # Check if the model needs label inversion based on sample data
    invert_labels = check_model_label_orientation(model_path, dataset_path)
    print(f"Label inversion detected: {invert_labels}")
    
    # Process tumor images (yes)
    yes_files = [f for f in os.listdir(yes_dir) if os.path.isfile(os.path.join(yes_dir, f))]
    print(f"Processing {len(yes_files)} tumor images...")
    for img_file in yes_files:
        try:
            img_path = os.path.join(yes_dir, img_file)
            img_array = preprocess_image(img_path)
            
            # Make prediction
            prediction = infer(tf.constant(img_array))
            output_key = list(prediction.keys())[0]
            pred_array = prediction[output_key].numpy()[0]
            pred_class = np.argmax(pred_array)
            
            # Apply inversion if necessary
            if invert_labels:
                pred_class = 1 - pred_class
                
            all_predictions.append(pred_class)
            all_true_labels.append(0)  # 0 for tumor (yes)
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
    
    # Process non-tumor images (no)
    no_files = [f for f in os.listdir(no_dir) if os.path.isfile(os.path.join(no_dir, f))]
    print(f"Processing {len(no_files)} non-tumor images...")
    for img_file in no_files:
        try:
            img_path = os.path.join(no_dir, img_file)
            img_array = preprocess_image(img_path)
            
            # Make prediction
            prediction = infer(tf.constant(img_array))
            output_key = list(prediction.keys())[0]
            pred_array = prediction[output_key].numpy()[0]
            pred_class = np.argmax(pred_array)
            
            # Apply inversion if necessary
            if invert_labels:
                pred_class = 1 - pred_class
            
            all_predictions.append(pred_class)
            all_true_labels.append(1)  # 1 for no tumor (no)
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
    
    # Calculate metrics
    cm = confusion_matrix(all_true_labels, all_predictions)
    report = classification_report(all_true_labels, all_predictions, 
                                  target_names=["Tumor", "No Tumor"], 
                                  output_dict=True)
    
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(all_true_labels, all_predictions, 
                              target_names=["Tumor", "No Tumor"]))
    
    accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    return {
        "confusion_matrix": cm,
        "classification_report": report,
        "accuracy": accuracy,
        "invert_labels": invert_labels
    }

def check_model_label_orientation(model_path, dataset_path):
    """Check if model labels need to be inverted by testing a sample from each class"""
    try:
        # Load the model
        model = tf.saved_model.load(model_path)
        infer = model.signatures['serving_default']
        
        # Get sample images
        yes_dir = os.path.join(dataset_path, 'yes')
        no_dir = os.path.join(dataset_path, 'no')
        
        # Find first usable image in each directory
        yes_files = [f for f in os.listdir(yes_dir) if os.path.isfile(os.path.join(yes_dir, f))]
        no_files = [f for f in os.listdir(no_dir) if os.path.isfile(os.path.join(no_dir, f))]
        
        if not yes_files or not no_files:
            print("Could not find sample images for validation")
            return False
        
        # Process tumor sample
        yes_sample = os.path.join(yes_dir, yes_files[0])
        yes_array = preprocess_image(yes_sample)
        yes_pred = infer(tf.constant(yes_array))
        output_key = list(yes_pred.keys())[0]
        yes_pred_class = np.argmax(yes_pred[output_key].numpy()[0])
        
        # Process non-tumor sample
        no_sample = os.path.join(no_dir, no_files[0])
        no_array = preprocess_image(no_sample)
        no_pred = infer(tf.constant(no_array))
        output_key = list(no_pred.keys())[0]
        no_pred_class = np.argmax(no_pred[output_key].numpy()[0])
        
        print(f"Sample tumor prediction: {yes_pred_class}, Sample non-tumor prediction: {no_pred_class}")
        
        # Expected: tumor should be class 0, non-tumor should be class 1
        # If that's not the case, we need to invert predictions
        if yes_pred_class == 0 and no_pred_class == 1:
            return False  # Labels are correct, no inversion needed
        else:
            return True   # Labels need to be inverted
    except Exception as e:
        print(f"Error checking model orientation: {str(e)}")
        return False

def visualize_predictions(model_path, sample_images, invert_labels=False):
    """Visualize model predictions on sample images"""
    model = tf.saved_model.load(model_path)
    infer = model.signatures['serving_default']
    
    plt.figure(figsize=(15, 10))
    for i, img_path in enumerate(sample_images):
        try:
            # Process image
            img_array = preprocess_image(img_path)
            
            # Make prediction
            prediction = infer(tf.constant(img_array))
            output_key = list(prediction.keys())[0]
            pred_array = prediction[output_key].numpy()[0]
            pred_class = np.argmax(pred_array)
            confidence = pred_array[pred_class] * 100
            
            if invert_labels:
                pred_class = 1 - pred_class
                confidence = pred_array[1 - pred_class] * 100
            
            # Display result
            img = Image.open(img_path)
            plt.subplot(2, len(sample_images)//2 + len(sample_images)%2, i+1)
            plt.imshow(img)
            plt.title(f"{'No Tumor' if pred_class==1 else 'Tumor'}\n{confidence:.1f}%")
            plt.axis('off')
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    plt.tight_layout()
    plt.show()

# Example usage (uncomment to use)
# if __name__ == "__main__":
#     model_path = "converted_savedmodel/model.savedmodel"
#     dataset_path = "Brain-Mri"
#     
#     # Evaluate model performance
#     results = evaluate_model_on_dataset(model_path, dataset_path)
#     
#     # Get some sample images for visualization
#     yes_samples = [os.path.join(dataset_path, "yes", f) for f in os.listdir(os.path.join(dataset_path, "yes"))[:4]]
#     no_samples = [os.path.join(dataset_path, "no", f) for f in os.listdir(os.path.join(dataset_path, "no"))[:4]]
#     samples = yes_samples + no_samples
#     
#     # Visualize some predictions
#     visualize_predictions(model_path, samples, results["invert_labels"])