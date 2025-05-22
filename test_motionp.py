import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from models.motip import build
from models.runtime_tracker import RuntimeTracker

def create_dummy_config():
    """Create a minimal configuration for testing"""
    return {
        "BACKBONE": "resnet50",
        "LR": 1e-4,
        "LR_BACKBONE_SCALE": 0.1,
        "DILATION": False,
        "NUM_CLASSES": 1,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "DETR_NUM_QUERIES": 300,
        "DETR_NUM_FEATURE_LEVELS": 4,
        "DETR_AUX_LOSS": True,
        "DETR_WITH_BOX_REFINE": True,
        "DETR_TWO_STAGE": False,
        "DETR_HIDDEN_DIM": 256,
        "DETR_MASKS": False,
        "DETR_POSITION_EMBEDDING": "sine",
        "DETR_NUM_HEADS": 8,
        "DETR_ENC_LAYERS": 6,
        "DETR_DEC_LAYERS": 6,
        "DETR_DIM_FEEDFORWARD": 1024,
        "DETR_DROPOUT": 0.1,
        "DETR_DEC_N_POINTS": 4,
        "DETR_ENC_N_POINTS": 4,
        "DETR_CLS_LOSS_COEF": 2.0,
        "DETR_BBOX_LOSS_COEF": 5.0,
        "DETR_GIOU_LOSS_COEF": 2.0,
        "DETR_FOCAL_ALPHA": 0.25,
        "DETR_SET_COST_CLASS": 2.0,
        "DETR_SET_COST_BBOX": 5.0,
        "DETR_SET_COST_GIOU": 2.0,
        "DETR_FRAMEWORK": "deformable_detr",
        "ONLY_DETR": False,
        "FFN_DIM_RATIO": 4,
        "FEATURE_DIM": 256,
        "MOTION_DIM": 32,
        "ID_DIM": 256,
        "NUM_ID_DECODER_LAYERS": 6,
        "HEAD_DIM": 32,
        "NUM_ID_VOCABULARY": 1000,
        "REL_PE_LENGTH": 100,
        "USE_AUX_LOSS": True,
        "USE_SHARED_AUX_HEAD": True,
        "MOTION_WEIGHT": 0.35
    }

def test_motionp():
    """Test MOTIonP implementation with a real video from DanceTrack"""
    # Create model
    print("Creating model...")
    config = create_dummy_config()
    model, criterion = build(config)
    model = model.to(config["DEVICE"])
    model.eval()
    
    # Create tracker
    print("Creating tracker...")
    tracker = RuntimeTracker(
        model=model,
        sequence_hw=(1080, 1920),  # DanceTrack videos are typically 1080p
        use_sigmoid=True,
        assignment_protocol="hungarian",
        miss_tolerance=30,
        det_thresh=0.5,
        newborn_thresh=0.5,
        id_thresh=0.1,
        area_thresh=0,
        only_detr=False,
        dtype=torch.float32
    )
    
    # Process frames
    print("Processing frames...")
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Path to the video frames
    video_path = "datasets/dancetrack/train/dance_001/img1"
    
    # Get all frame files
    frame_files = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
    
    for frame_file in frame_files:
        # Load and preprocess frame
        frame_path = os.path.join(video_path, frame_file)
        img = Image.open(frame_path)
        img_tensor = transform(img).unsqueeze(0).to(config["DEVICE"])
        
        # Forward pass
        with torch.no_grad():
            outputs = model(img_tensor)
            
            # Update tracker
            tracker.update(outputs)
            
            # Get tracking results
            track_results = tracker.get_track_results()
            
            print(f"Frame {frame_file}: Found {len(track_results)} tracks")
            
            # Print track information
            for track_id, track_info in track_results.items():
                print(f"  Track {track_id}: {track_info}")

if __name__ == "__main__":
    test_motionp() 