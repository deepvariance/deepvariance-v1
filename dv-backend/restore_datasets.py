"""
Script to restore datasets from the data directory to the database
"""
import os
from database import DatasetDB
from datetime import datetime

def restore_datasets():
    data_dir = 'data'
    
    if not os.path.exists(data_dir):
        print(f"✗ Data directory '{data_dir}' not found")
        return
    
    print("Scanning data directory for datasets...\n")
    
    # Find all dataset directories
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        
        if os.path.isdir(item_path) and not item.startswith('.'):
            # Try to determine if it's a valid dataset
            test_path = os.path.join(item_path, 'test')
            train_path = os.path.join(item_path, 'train')
            
            if os.path.exists(test_path):
                # This looks like an image classification dataset
                classes = [d for d in os.listdir(test_path) 
                          if os.path.isdir(os.path.join(test_path, d)) and not d.startswith('.')]
                num_classes = len(classes)
                
                # Count total samples
                total_samples = 0
                for class_name in classes:
                    class_path = os.path.join(test_path, class_name)
                    if os.path.exists(class_path):
                        samples = len([f for f in os.listdir(class_path) 
                                      if os.path.isfile(os.path.join(class_path, f)) and not f.startswith('.')])
                        total_samples += samples
                
                # Determine dataset name
                dataset_name = 'Lung Colon Cancer' if num_classes == 5 else f'Dataset {item[:8]}'
                
                # Create dataset entry
                dataset_data = {
                    'id': item,  # Use existing UUID as ID
                    'name': dataset_name,
                    'domain': 'vision',  # Image datasets are vision domain
                    'description': f'{num_classes} classes, {total_samples} samples',
                    'dataset_type': 'image',
                    'file_path': item_path,
                    'size': total_samples,
                    'num_samples': total_samples,
                    'total_samples': total_samples,
                    'num_features': None,
                    'num_classes': num_classes,
                    'task_type': 'classification',
                    'supported_tasks': ['classification'],
                    'readiness': 'ready',
                    'created_at': datetime.now().isoformat()
                }
                
                try:
                    DatasetDB.create(dataset_data)
                    print(f'✓ Registered: {dataset_name}')
                    print(f'  - ID: {item}')
                    print(f'  - Classes: {num_classes}')
                    print(f'  - Samples: {total_samples}')
                    print(f'  - Path: {item_path}')
                    print()
                except Exception as e:
                    print(f'✗ Failed to register {item}: {e}')
                    print()

if __name__ == '__main__':
    restore_datasets()
    print("\n✓ Dataset restoration complete!")
