import streamlit as st
import numpy as np
import time
from PIL import Image
import cv2
import torch
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configuration de la page
st.set_page_config(
    page_title="Détection des Dommages Routiers",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ============ CONFIGURATION DES MODÈLES ============
# MODIFIEZ CES CHEMINS SELON VOS FICHIERS
MODEL_CONFIG = {
    'unet': {
        'path': 'best_unet_model.pth',  # Mettez le chemin complet de votre modèle U-Net
        'img_size': 640,
        'num_classes': 6,
        'encoder': 'resnet34'
    },
    'yolo': {
        'path': 'best.pt',  # Mettez le chemin complet de votre modèle YOLO
        'img_size': 640,
        'conf': 0.25
    },
    'hybrid': {
        'path': 'best_hybrid_model.pth',  # Mettez le chemin complet de votre modèle Hybride
        'img_size': 640,
        'num_classes': 6,
        'encoder': 'resnet34'
    }
}

# Classes et couleurs
CLASS_NAMES_UNET = [
    'Background',
    'Longitudinal_crack',
    'Transverse_crack',
    'Alligator_crack',
    'Other_damage',
    'Pothole'
]

CLASS_NAMES_YOLO = [
    'Longitudinal_crack',
    'Longitudinal_crack',
    'Transverse_crack',
    'Alligator_crack',
    'Other_damage',
    'Pothole'
]

COLORS_UNET = np.array([
    [0, 0, 0],       # Background
    [255, 0, 0],     # Longitudinal crack - Red
    [0, 255, 0],     # Transverse crack - Green
    [0, 0, 255],     # Alligator crack - Blue
    [255, 0, 255],    # Other damage - Magenta
    [255, 255, 0]   # Pothole - Yellow
], dtype=np.uint8)

# ============ FONCTIONS UTILITAIRES ============

# ============ FONCTIONS UTILITAIRES ============

def calculate_metrics(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Calcule IoU et F1-score entre prédictions et ground truth
    """
    if not gt_boxes or not pred_boxes:
        return {'iou': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    def calculate_box_iou(box1, box2):
        """Calcule l'IoU entre deux bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        inter_width = max(0, inter_xmax - inter_xmin)
        inter_height = max(0, inter_ymax - inter_ymin)
        inter_area = inter_width * inter_height
        
        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    # Calculer IoU moyen
    ious = []
    matched_pred = set()
    matched_gt = set()
    
    for gt_idx, gt_box in enumerate(gt_boxes):
        best_iou = 0
        best_pred_idx = -1
        
        for pred_idx, pred_box in enumerate(pred_boxes):
            if pred_idx in matched_pred:
                continue
            
            iou = calculate_box_iou(gt_box['bbox'], pred_box['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = pred_idx
        
        if best_iou > iou_threshold:
            ious.append(best_iou)
            matched_pred.add(best_pred_idx)
            matched_gt.add(gt_idx)
    
    # Métriques
    mean_iou = np.mean(ious) if ious else 0.0
    
    true_positives = len(matched_gt)
    false_positives = len(pred_boxes) - len(matched_pred)
    false_negatives = len(gt_boxes) - len(matched_gt)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'iou': mean_iou,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'tp': true_positives,
        'fp': false_positives,
        'fn': false_negatives
    }

def parse_yolo_label(label_path, img_width, img_height):
    """
    Parse un fichier label YOLO format
    Chaque ligne: class_id x_center y_center width height (normalisé 0-1)
    Retourne une liste de bounding boxes
    """
    boxes = []
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convertir de normalisé à pixels
                x_center_px = x_center * img_width
                y_center_px = y_center * img_height
                width_px = width * img_width
                height_px = height * img_height
                
                # Calculer les coins de la boîte
                x1 = int(x_center_px - width_px / 2)
                y1 = int(y_center_px - height_px / 2)
                x2 = int(x_center_px + width_px / 2)
                y2 = int(y_center_px + height_px / 2)
                
                boxes.append({
                    'class_id': class_id,
                    'bbox': (x1, y1, x2, y2),
                    'center': (x_center_px, y_center_px),
                    'size': (width_px, height_px)
                })
        
        return boxes
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier label: {e}")
        return []

def draw_ground_truth_boxes(image, boxes, class_names):
    """
    Dessine les bounding boxes véritables sur l'image
    """
    img_with_boxes = np.array(image).copy()
    
    # Couleurs pour chaque classe (BGR pour OpenCV)
    colors = {
        0: (255, 0, 0),      # Rouge
        1: (0, 255, 0),      # Vert
        2: (0, 0, 255),      # Bleu
        3: (255, 255, 0),    # Cyan
        4: (255, 0, 255),    # Magenta
        5: (0, 255, 255),    # Jaune
    }
    
    for box in boxes:
        class_id = box['class_id']
        x1, y1, x2, y2 = box['bbox']
        
        # Couleur de la classe
        color = colors.get(class_id, (255, 255, 255))
        
        # Dessiner le rectangle
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 3)
        
        # Ajouter le label
        if class_id < len(class_names):
            label = class_names[class_id]
        else:
            label = f"Class {class_id}"
        
        # Fond pour le texte
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_with_boxes, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
        
        # Texte
        cv2.putText(img_with_boxes, label, (x1 + 5, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return Image.fromarray(img_with_boxes)

# ============ FONCTIONS DE CHARGEMENT DES MODÈLES ============

@st.cache_resource
def load_unet_model(model_path, device):
    """Charge le modèle U-Net"""
    try:
        # Vérifier si le fichier existe
        if not Path(model_path).exists():
            st.error(f"❌ Fichier U-Net introuvable: {model_path}")
            st.info("💡 Veuillez uploader votre modèle ou vérifier le chemin dans la sidebar")
            return None
            
        import segmentation_models_pytorch as smp
        
        model = smp.Unet(
            encoder_name=MODEL_CONFIG['unet']['encoder'],
            encoder_weights=None,
            in_channels=3,
            classes=MODEL_CONFIG['unet']['num_classes'],
            activation=None
        )
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            st.success(f"✅ Modèle U-Net chargé (mIoU: {checkpoint.get('iou', 'N/A')})")
        else:
            model.load_state_dict(checkpoint)
            st.success("✅ Modèle U-Net chargé avec succès")
        
        model = model.to(device)
        model.eval()
        
        return model
    except ImportError:
        st.error("❌ Erreur: segmentation_models_pytorch non installé!")
        st.code("pip install segmentation-models-pytorch", language="bash")
        return None
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle U-Net: {e}")
        return None

@st.cache_resource
def load_yolo_model(model_path):
    """Charge le modèle YOLO"""
    try:
        # Vérifier si le fichier existe
        if not Path(model_path).exists():
            st.error(f"❌ Fichier YOLO introuvable: {model_path}")
            st.info("💡 Veuillez uploader votre modèle ou vérifier le chemin dans la sidebar")
            return None
            
        from ultralytics import YOLO
        model = YOLO(model_path)
        st.success("✅ Modèle YOLO chargé avec succès")
        return model
    except ImportError:
        st.error("❌ Erreur: ultralytics non installé!")
        st.code("pip install ultralytics", language="bash")
        return None
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle YOLO: {e}")
        return None

@st.cache_resource
def load_hybrid_model(model_path, device):
    """Charge le modèle Hybride (PAN architecture)"""
    try:
        # Vérifier si le fichier existe
        if not Path(model_path).exists():
            st.error(f"❌ Fichier Hybride introuvable: {model_path}")
            st.info("💡 Veuillez uploader votre modèle ou vérifier le chemin dans la sidebar")
            return None
            
        import segmentation_models_pytorch as smp
        
        # Charger le checkpoint pour inspecter sa structure
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Extraire les poids du modèle
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Détecter l'architecture en inspectant les clés
        keys = list(state_dict.keys())
        
        # Déterminer l'architecture basée sur les clés
        model = None
        loaded_arch = None
        
        # Liste des architectures à essayer avec leurs noms
        architectures_to_try = [
            ('PAN', smp.PAN),  # Votre modèle hybride utilise PAN !
            ('Unet', smp.Unet),
            ('UnetPlusPlus', smp.UnetPlusPlus),
            ('FPN', smp.FPN),
            ('PSPNet', smp.PSPNet),
            ('DeepLabV3', smp.DeepLabV3),
            ('DeepLabV3Plus', smp.DeepLabV3Plus),
            ('MAnet', smp.MAnet),
            ('Linknet', smp.Linknet),
        ]
        
        # Essayer chaque architecture
        for arch_name, arch_class in architectures_to_try:
            try:
                test_model = arch_class(
                    encoder_name=MODEL_CONFIG['hybrid']['encoder'],
                    encoder_weights=None,
                    in_channels=3,
                    classes=MODEL_CONFIG['hybrid']['num_classes'],
                    activation=None
                )
                
                # Essayer de charger les poids
                test_model.load_state_dict(state_dict, strict=True)
                
                # Si ça marche, on a trouvé la bonne architecture
                model = test_model
                loaded_arch = arch_name
                break
                
            except Exception as e:
                # Cette architecture ne correspond pas, essayer la suivante
                continue
        
        if model is None:
            st.error("❌ Impossible de charger le modèle Hybride automatiquement")
            st.info("💡 Le modèle devrait utiliser l'architecture PAN (Pyramid Attention Network)")
            
            # Afficher les clés pour debug
            with st.expander("🔍 Structure du modèle détectée"):
                st.write("Premières clés :")
                st.code('\n'.join(keys[:20]))
                
                # Analyser la structure
                encoder_keys = [k for k in keys if k.startswith('encoder.')]
                decoder_keys = [k for k in keys if k.startswith('decoder.')]
                
                st.write(f"- Clés encoder: {len(encoder_keys)}")
                st.write(f"- Clés decoder: {len(decoder_keys)}")
            
            return None
        
        model = model.to(device)
        model.eval()
        
        if isinstance(checkpoint, dict) and 'iou' in checkpoint:
            st.success(f"✅ Modèle Hybride chargé ({loaded_arch}) - mIoU: {checkpoint['iou']:.4f}")
        else:
            st.success(f"✅ Modèle Hybride chargé ({loaded_arch})")
        
        return model
        
    except ImportError:
        st.error("❌ Erreur: segmentation_models_pytorch non installé!")
        st.code("pip install segmentation-models-pytorch", language="bash")
        return None
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle Hybride: {e}")
        
        with st.expander("🔍 Détails de l'erreur"):
            import traceback
            st.code(traceback.format_exc())
        
        return None

# ============ FONCTIONS DE PRÉDICTION ============

def predict_unet(image, model, device):
    """Prédiction avec U-Net"""
    start_time = time.time()
    
    # Prétraitement
    img_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    
    transform = A.Compose([
        A.Resize(MODEL_CONFIG['unet']['img_size'], MODEL_CONFIG['unet']['img_size']),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    transformed = transform(image=img_rgb)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Prédiction
    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0]
    
    # Créer l'image segmentée colorée
    colored_mask = COLORS_UNET[pred_mask]
    
    # Overlay
    img_resized = cv2.resize(img_rgb, (MODEL_CONFIG['unet']['img_size'], MODEL_CONFIG['unet']['img_size']))
    alpha = 0.5
    overlay = cv2.addWeighted(img_resized, 1-alpha, colored_mask, alpha, 0)
    segmented_img = Image.fromarray(overlay)
    
    # Analyser les prédictions et créer des bounding boxes approximatives
    predictions = {}
    pred_boxes = []
    
    unique, counts = np.unique(pred_mask, return_counts=True)
    total_pixels = pred_mask.size
    
    for class_id, count in zip(unique, counts):
        if class_id > 0:  # Ignorer le background
            percentage = (count / total_pixels)
            if percentage > 0.001:  # Au moins 0.1%
                predictions[class_id] = percentage
                
                # Créer une bounding box approximative pour cette classe
                mask_class = (pred_mask == class_id).astype(np.uint8)
                contours, _ = cv2.findContours(mask_class, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    if cv2.contourArea(contour) > 100:  # Filtrer petites régions
                        x, y, w, h = cv2.boundingRect(contour)
                        # Convertir à l'échelle de l'image originale
                        scale_x = image.size[0] / MODEL_CONFIG['unet']['img_size']
                        scale_y = image.size[1] / MODEL_CONFIG['unet']['img_size']
                        pred_boxes.append({
                            'class_id': class_id - 1,  # Ajuster pour correspondre à YOLO (pas de background)
                            'bbox': (int(x*scale_x), int(y*scale_y), int((x+w)*scale_x), int((y+h)*scale_y)),
                            'confidence': percentage
                        })
    
    execution_time = time.time() - start_time
    
    return segmented_img, predictions, execution_time, pred_boxes

def predict_yolo(image, model):
    """Prédiction avec YOLO"""
    start_time = time.time()
    
    # Convertir PIL en numpy et redimensionner
    img_np = np.array(image)
    img_resized = cv2.resize(img_np, (MODEL_CONFIG['yolo']['img_size'], MODEL_CONFIG['yolo']['img_size']))
    
    # Prédiction
    results = model.predict(
        img_resized,
        conf=MODEL_CONFIG['yolo']['conf'],
        imgsz=MODEL_CONFIG['yolo']['img_size'],
        verbose=False
    )
    
    # Créer l'image avec les boîtes
    result_img = results[0].plot()
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    segmented_img = Image.fromarray(result_img)
    
    # Analyser les détections et créer les boxes
    predictions = {}
    pred_boxes = []
    
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        scale_x = image.size[0] / MODEL_CONFIG['yolo']['img_size']
        scale_y = image.size[1] / MODEL_CONFIG['yolo']['img_size']
        
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            # Convertir à l'échelle de l'image originale
            x1 = int(xyxy[0] * scale_x)
            y1 = int(xyxy[1] * scale_y)
            x2 = int(xyxy[2] * scale_x)
            y2 = int(xyxy[3] * scale_y)
            
            pred_boxes.append({
                'class_id': cls_id,
                'bbox': (x1, y1, x2, y2),
                'confidence': conf
            })
            
            if cls_id not in predictions:
                predictions[cls_id] = conf
            else:
                predictions[cls_id] = max(predictions[cls_id], conf)
    
    execution_time = time.time() - start_time
    
    return segmented_img, predictions, execution_time, pred_boxes

def predict_hybrid(image, hybrid_model, device):
    """Prédiction avec le modèle Hybride"""
    start_time = time.time()
    
    # Si un modèle hybride spécifique existe, l'utiliser
    if hybrid_model is not None:
        # Même prétraitement que U-Net
        img_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        
        transform = A.Compose([
            A.Resize(MODEL_CONFIG['hybrid']['img_size'], MODEL_CONFIG['hybrid']['img_size']),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        transformed = transform(image=img_rgb)
        image_tensor = transformed['image'].unsqueeze(0).to(device)
        
        # Prédiction
        with torch.no_grad():
            output = hybrid_model(image_tensor)
            pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0]
        
        # Créer l'image segmentée colorée
        colored_mask = COLORS_UNET[pred_mask]
        
        # Overlay
        img_resized = cv2.resize(img_rgb, (MODEL_CONFIG['hybrid']['img_size'], MODEL_CONFIG['hybrid']['img_size']))
        alpha = 0.5
        overlay = cv2.addWeighted(img_resized, 1-alpha, colored_mask, alpha, 0)
        segmented_img = Image.fromarray(overlay)
        
        # Analyser les prédictions et créer des bounding boxes
        predictions = {}
        pred_boxes = []
        
        unique, counts = np.unique(pred_mask, return_counts=True)
        total_pixels = pred_mask.size
        
        for class_id, count in zip(unique, counts):
            if class_id > 0:  # Ignorer le background
                percentage = (count / total_pixels)
                if percentage > 0.001:  # Au moins 0.1%
                    predictions[class_id] = percentage
                    
                    # Créer une bounding box approximative
                    mask_class = (pred_mask == class_id).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_class, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        if cv2.contourArea(contour) > 100:
                            x, y, w, h = cv2.boundingRect(contour)
                            scale_x = image.size[0] / MODEL_CONFIG['hybrid']['img_size']
                            scale_y = image.size[1] / MODEL_CONFIG['hybrid']['img_size']
                            pred_boxes.append({
                                'class_id': class_id - 1,
                                'bbox': (int(x*scale_x), int(y*scale_y), int((x+w)*scale_x), int((y+h)*scale_y)),
                                'confidence': percentage
                            })
        
        execution_time = time.time() - start_time
        
        return segmented_img, predictions, execution_time, pred_boxes
    else:
        st.warning("⚠️ Modèle Hybride non disponible. Veuillez l'uploader dans la sidebar.")
        return None, {}, 0.0, []

# ============ FONCTIONS D'AFFICHAGE ============

def create_comparison_chart(results):
    """Crée un graphique de comparaison des temps d'exécution"""
    models = list(results.keys())
    times = [results[m]['time'] for m in models]
    
    model_names = {'unet': 'U-Net Custom', 'yolo': 'YOLO Prédéfini', 'hybrid': 'Architecture Hybride'}
    display_names = [model_names.get(m, m) for m in models]
    
    fig = go.Figure(data=[
        go.Bar(
            x=display_names,
            y=times,
            marker_color=['#9b59b6', '#e74c3c', '#27ae60'],
            text=[f"{t:.3f}s" for t in times],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Temps d'exécution par modèle",
        xaxis_title="Modèle",
        yaxis_title="Temps (secondes)",
        height=400,
        showlegend=False
    )
    
    return fig

def create_confidence_chart(results):
    """Crée un graphique de comparaison des confiances"""
    data = []
    
    for model, res in results.items():
        model_names = {'unet': 'U-Net', 'yolo': 'YOLO', 'hybrid': 'Hybride'}
        for class_id, conf in res['predictions'].items():
            class_name = CLASS_NAMES_UNET[class_id] if model == 'unet' or model == 'hybrid' else CLASS_NAMES_YOLO[class_id]
            data.append({
                'Modèle': model_names.get(model, model),
                'Type de dommage': class_name,
                'Confiance': conf * 100
            })
    
    if not data:
        return None
    
    import pandas as pd
    df = pd.DataFrame(data)
    
  

# ============ INTERFACE PRINCIPALE ============

def main():
    # En-tête
    st.markdown("""
        <h1 style='text-align: center; color: #2c3e50;'>
            🛣️ Système de Détection des Dommages Routiers
        </h1>
        <p style='text-align: center; color: #7f8c8d; font-size: 18px;'>
            Analyse intelligente des infrastructures routières - Projet RDD2022
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📋 À propos du projet")
        st.info("""
        Ce système utilise des techniques avancées de Deep Learning pour :
        - 🔍 Détecter les dommages routiers
        - 🎯 Segmenter les zones endommagées
        - 📊 Classifier les types de dégradation
        """)
        
        st.markdown("---")
        st.markdown("### ⚙️ Configuration des modèles")
        
        st.warning("⚠️ Veuillez configurer les chemins de vos modèles entraînés")
        
        # Configuration avancée pour le modèle Hybride
        with st.expander("🔧 Configuration Hybride (Avancée)"):
            st.info("Votre modèle Hybride utilise l'architecture PAN (Pyramid Attention Network)")
            
            hybrid_architectures = ['Auto-detect', 'PAN', 'Unet', 'UnetPlusPlus', 'FPN', 'PSPNet', 'DeepLabV3', 'DeepLabV3Plus']
            selected_arch = st.selectbox(
                "Architecture du modèle Hybride",
                hybrid_architectures,
                index=0  # Auto-detect par défaut
            )
            
            if selected_arch != 'Auto-detect':
                MODEL_CONFIG['hybrid']['architecture'] = selected_arch
            
            hybrid_encoders = ['resnet34', 'resnet50', 'efficientnet-b0', 'mobilenet_v2', 'vgg16']
            selected_encoder = st.selectbox(
                "Encoder du modèle Hybride",
                hybrid_encoders,
                index=0
            )
            MODEL_CONFIG['hybrid']['encoder'] = selected_encoder
            
            st.caption("💡 PAN combine les avantages de la pyramide d'attention et de la segmentation multi-échelle")
        
        # Upload de fichiers ou saisie de chemins
        use_upload = st.checkbox("📤 Uploader les modèles", value=False)
        
        if use_upload:
            st.markdown("#### Upload U-Net")
            unet_file = st.file_uploader("Modèle U-Net (.pth)", type=['pth', 'pt'], key="unet_upload")
            if unet_file:
                # Sauvegarder temporairement
                unet_temp_path = f"temp_unet_{unet_file.name}"
                with open(unet_temp_path, "wb") as f:
                    f.write(unet_file.read())
                MODEL_CONFIG['unet']['path'] = unet_temp_path
                st.success(f"✅ U-Net uploadé: {unet_file.name}")
            
            st.markdown("#### Upload YOLO")
            yolo_file = st.file_uploader("Modèle YOLO (.pt)", type=['pt'], key="yolo_upload")
            if yolo_file:
                # Sauvegarder temporairement
                yolo_temp_path = f"temp_yolo_{yolo_file.name}"
                with open(yolo_temp_path, "wb") as f:
                    f.write(yolo_file.read())
                MODEL_CONFIG['yolo']['path'] = yolo_temp_path
                st.success(f"✅ YOLO uploadé: {yolo_file.name}")
            
            st.markdown("#### Upload Hybride")
            hybrid_file = st.file_uploader("Modèle Hybride (.pth)", type=['pth', 'pt'], key="hybrid_upload")
            if hybrid_file:
                # Sauvegarder temporairement
                hybrid_temp_path = f"temp_hybrid_{hybrid_file.name}"
                with open(hybrid_temp_path, "wb") as f:
                    f.write(hybrid_file.read())
                MODEL_CONFIG['hybrid']['path'] = hybrid_temp_path
                st.success(f"✅ Hybride uploadé: {hybrid_file.name}")
        else:
            # Saisie manuelle des chemins
            unet_path = st.text_input(
                "📂 Chemin U-Net", 
                MODEL_CONFIG['unet']['path'],
                help="Exemple: /content/drive/MyDrive/unet_models/unet_best.pth"
            )
            yolo_path = st.text_input(
                "📂 Chemin YOLO", 
                MODEL_CONFIG['yolo']['path'],
                help="Exemple: /content/drive/MyDrive/yolov8_backup/best.pt"
            )
            hybrid_path = st.text_input(
                "📂 Chemin Hybride", 
                MODEL_CONFIG['hybrid']['path'],
                help="Exemple: /content/drive/MyDrive/hybrid_models/hybrid_best.pth"
            )
            
            MODEL_CONFIG['unet']['path'] = unet_path
            MODEL_CONFIG['yolo']['path'] = yolo_path
            MODEL_CONFIG['hybrid']['path'] = hybrid_path
            
            # Vérifier l'existence des fichiers
            col1, col2, col3 = st.columns(3)
            with col1:
                if Path(unet_path).exists():
                    st.success("✅ U-Net")
                else:
                    st.error("❌ U-Net")
            with col2:
                if Path(yolo_path).exists():
                    st.success("✅ YOLO")
                else:
                    st.error("❌ YOLO")
            with col3:
                if Path(hybrid_path).exists():
                    st.success("✅ Hybride")
                else:
                    st.error("❌ Hybride")
        
        st.markdown("---")
        st.markdown("### 🎨 Légende des dommages (U-Net)")
        for class_id, class_name in enumerate(CLASS_NAMES_UNET):
            if class_id > 0:  # Ignorer background
                color = COLORS_UNET[class_id]
                color_hex = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
                st.markdown(
                    f"<span style='color: {color_hex}; font-size: 20px;'>⬤</span> {class_name}",
                    unsafe_allow_html=True
                )
        
        st.markdown("---")
        st.markdown("### 💡 Aide")
        with st.expander("Comment utiliser ?"):
            st.markdown("""
            1. **Configurez les chemins** de vos modèles dans la sidebar
            2. **Uploadez une image** de route
            3. **Choisissez un modèle** (U-Net, YOLO ou Hybride)
            4. **Lancez l'analyse** pour voir les résultats
            5. **Comparez les modèles** dans l'onglet comparaison
            
            **Exemples de chemins:**
            - Google Drive: `/content/drive/MyDrive/models/best.pt`
            - Local: `./outputs/unet/best_model.pth`
            
            **Format du fichier label:**
            ```
            class_id x_center y_center width height
            0 0.5 0.3 0.1 0.15
            2 0.7 0.6 0.08 0.12
            ```
            Toutes les valeurs sont normalisées (0 à 1)
            """)
    
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        st.success(f"🚀 GPU détecté: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("⚠️ CPU utilisé (plus lent). Utilisez un GPU pour de meilleures performances.")
    
    # Zone de chargement d'image
    uploaded_file = st.file_uploader(
        "📁 Chargez une image de route",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Formats supportés: JPG, JPEG, PNG, BMP"
    )
    
    # Zone de chargement du label (optionnel)
    st.markdown("### 📋 Label Ground Truth (Optionnel)")
    uploaded_label = st.file_uploader(
        "📄 Chargez le fichier label (.txt)",
        type=['txt'],
        help="Format YOLO: class_id x_center y_center width height (normalisé 0-1)",
        key="label_upload"
    )
    
    if uploaded_label:
        st.success("✅ Fichier label chargé ! Les bounding boxes véritables seront affichées.")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Parser le label si disponible
        ground_truth_boxes = []
        if uploaded_label is not None:
            # Sauvegarder temporairement le fichier label
            label_temp_path = f"temp_label_{uploaded_label.name}"
            with open(label_temp_path, "wb") as f:
                f.write(uploaded_label.read())
            
            # Parser les boxes
            ground_truth_boxes = parse_yolo_label(label_temp_path, image.size[0], image.size[1])
            
            if ground_truth_boxes:
                st.info(f"📦 {len(ground_truth_boxes)} dommage(s) détecté(s) dans le label")
        
        st.markdown("---")
        
        # Afficher l'image originale avec ou sans boxes
        st.markdown("### 📷 Image Originale")
        
        if ground_truth_boxes:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Sans annotations**")
                st.image(image, use_container_width=True)
            
            with col2:
                st.markdown("**Avec Ground Truth**")
                # Dessiner les boxes véritables
                image_with_gt = draw_ground_truth_boxes(image, ground_truth_boxes, CLASS_NAMES_YOLO)
                st.image(image_with_gt, use_container_width=True)
                
                # Afficher les détails des boxes
                with st.expander("🔍 Détails des annotations"):
                    for idx, box in enumerate(ground_truth_boxes):
                        class_name = CLASS_NAMES_YOLO[box['class_id']] if box['class_id'] < len(CLASS_NAMES_YOLO) else f"Class {box['class_id']}"
                        st.markdown(f"**Box {idx+1}:** {class_name}")
                        st.caption(f"Position: ({box['bbox'][0]}, {box['bbox'][1]}) → ({box['bbox'][2]}, {box['bbox'][3]})")
        else:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption=f"Dimensions: {image.size[0]}x{image.size[1]} pixels", use_container_width=True)
        
        st.markdown("---")
        
        # Tabs pour les différentes analyses
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔷 U-Net Custom",
            "🔶 YOLO Prédéfini",
            "🔸 Architecture Hybride",
            "📊 Comparaison Globale"
        ])
        
        # Tab U-Net
        with tab1:
            st.markdown("### 🔷 Analyse avec U-Net Custom")
            st.info("Architecture U-Net implémentée from scratch avec ResNet34 encoder")
            
            if st.button("▶️ Exécuter U-Net", key="unet_btn", type="primary"):
                with st.spinner("Chargement du modèle U-Net..."):
                    unet_model = load_unet_model(MODEL_CONFIG['unet']['path'], device)
                
                if unet_model is not None:
                    with st.spinner("Traitement en cours avec U-Net..."):
                        segmented, predictions, exec_time, pred_boxes = predict_unet(image, unet_model, device)
                        
                        # Afficher les résultats (sans image originale)
                        if ground_truth_boxes:
                            cols_display = st.columns(2)
                            
                            with cols_display[0]:
                                st.markdown("**Ground Truth**")
                                image_with_gt = draw_ground_truth_boxes(image, ground_truth_boxes, CLASS_NAMES_YOLO)
                                st.image(image_with_gt, use_container_width=True)
                            
                            with cols_display[1]:
                                st.markdown("**Segmentation U-Net**")
                                st.image(segmented, use_container_width=True)
                        else:
                            st.markdown("**Segmentation U-Net**")
                            st.image(segmented, use_container_width=True)
                        
                        # Métriques
                        st.markdown("### 📊 Résultats")
                        
                        metric_cols = st.columns(4)
                        with metric_cols[0]:
                            st.metric("⏱️ Temps d'exécution", f"{exec_time:.3f}s")
                        with metric_cols[1]:
                            st.metric("🔍 Dommages détectés", len(predictions))
                        
                        # Calculer IoU et F1 si ground truth disponible
                        if ground_truth_boxes:
                            metrics = calculate_metrics(pred_boxes, ground_truth_boxes)
                            with metric_cols[2]:
                                st.metric("📐 IoU Moyen", f"{metrics['iou']:.3f}")
                            with metric_cols[3]:
                                st.metric("🎯 F1-Score", f"{metrics['f1']:.3f}")
                        else:
                            with metric_cols[2]:
                                st.metric("📏 Résolution", f"{image.size[0]}x{image.size[1]}")
                            with metric_cols[3]:
                                avg_conf = np.mean(list(predictions.values())) if predictions else 0
                                st.metric("💯 Surface moyenne", f"{avg_conf:.1%}")
                        
                        # Supprimer les détections détaillées
                        if predictions:
                            st.success(f"✅ {len(predictions)} type(s) de dommage détecté(s)")
                        else:
                            st.success("✅ Aucun dommage significatif détecté")
        
        # Tab YOLO
        with tab2:
            st.markdown("### 🔶 Analyse avec YOLO Prédéfini")
            st.info("Modèle YOLOv8 pré-entraîné et fine-tuné sur RDD2022")
            
            if st.button("▶️ Exécuter YOLO", key="yolo_btn", type="primary"):
                with st.spinner("Chargement du modèle YOLO..."):
                    yolo_model = load_yolo_model(MODEL_CONFIG['yolo']['path'])
                
                if yolo_model is not None:
                    with st.spinner("Traitement en cours avec YOLO..."):
                        segmented, predictions, exec_time, pred_boxes = predict_yolo(image, yolo_model)
                        
                        if ground_truth_boxes:
                            cols_display = st.columns(2)
                            
                            with cols_display[0]:
                                st.markdown("**Ground Truth**")
                                image_with_gt = draw_ground_truth_boxes(image, ground_truth_boxes, CLASS_NAMES_YOLO)
                                st.image(image_with_gt, use_container_width=True)
                            
                            with cols_display[1]:
                                st.markdown("**Détection YOLO**")
                                st.image(segmented, use_container_width=True)
                        else:
                            st.markdown("**Détection YOLO**")
                            st.image(segmented, use_container_width=True)
                        
                        st.markdown("### 📊 Résultats")
                        
                        metric_cols = st.columns(4)
                        with metric_cols[0]:
                            st.metric("⏱️ Temps d'exécution", f"{exec_time:.3f}s")
                        with metric_cols[1]:
                            st.metric("🔍 Dommages détectés", len(predictions))
                        
                        if ground_truth_boxes:
                            metrics = calculate_metrics(pred_boxes, ground_truth_boxes)
                            with metric_cols[2]:
                                st.metric("📐 IoU Moyen", f"{metrics['iou']:.3f}")
                            with metric_cols[3]:
                                st.metric("🎯 F1-Score", f"{metrics['f1']:.3f}")
                        else:
                            with metric_cols[2]:
                                st.metric("📏 Résolution", f"{image.size[0]}x{image.size[1]}")
                            with metric_cols[3]:
                                avg_conf = np.mean(list(predictions.values())) if predictions else 0
                                st.metric("💯 Confiance moyenne", f"{avg_conf:.1%}")
                        
                        if predictions:
                            st.success(f"✅ {len(predictions)} type(s) de dommage détecté(s)")
                        else:
                            st.success("✅ Aucun dommage détecté")
        
        # Tab Hybride
        with tab3:
            st.markdown("### 🔸 Analyse avec Architecture Hybride")
            st.info("Architecture PAN (Pyramid Attention Network) - Combine pyramide d'attention et segmentation multi-échelle")
            
            if st.button("▶️ Exécuter Hybride", key="hybrid_btn", type="primary"):
                with st.spinner("Chargement du modèle Hybride (PAN)..."):
                    hybrid_model = load_hybrid_model(MODEL_CONFIG['hybrid']['path'], device)
                
                if hybrid_model is not None:
                    with st.spinner("Traitement en cours avec le modèle hybride..."):
                        segmented, predictions, exec_time, pred_boxes = predict_hybrid(image, hybrid_model, device)
                        
                        if segmented is not None:
                            if ground_truth_boxes:
                                cols_display = st.columns(2)
                                
                                with cols_display[0]:
                                    st.markdown("**Ground Truth**")
                                    image_with_gt = draw_ground_truth_boxes(image, ground_truth_boxes, CLASS_NAMES_YOLO)
                                    st.image(image_with_gt, use_container_width=True)
                                
                                with cols_display[1]:
                                    st.markdown("**Prédiction Hybride**")
                                    st.image(segmented, use_container_width=True)
                            else:
                                st.markdown("**Prédiction Hybride**")
                                st.image(segmented, use_container_width=True)
                            
                            st.markdown("### 📊 Résultats")
                            
                            metric_cols = st.columns(4)
                            with metric_cols[0]:
                                st.metric("⏱️ Temps d'exécution", f"{exec_time:.3f}s")
                            with metric_cols[1]:
                                st.metric("🔍 Dommages détectés", len(predictions))
                            
                            if ground_truth_boxes:
                                metrics = calculate_metrics(pred_boxes, ground_truth_boxes)
                                with metric_cols[2]:
                                    st.metric("📐 IoU Moyen", f"{metrics['iou']:.3f}")
                                with metric_cols[3]:
                                    st.metric("🎯 F1-Score", f"{metrics['f1']:.3f}")
                            else:
                                with metric_cols[2]:
                                    st.metric("📏 Résolution", f"{image.size[0]}x{image.size[1]}")
                                with metric_cols[3]:
                                    avg_conf = np.mean(list(predictions.values())) if predictions else 0
                                    st.metric("💯 Surface moyenne", f"{avg_conf:.1%}")
                            
                            if predictions:
                                st.success(f"✅ {len(predictions)} type(s) de dommage détecté(s)")
                            else:
                                st.success("✅ Aucun dommage significatif détecté")
        
        # Tab Comparaison
        with tab4:
            st.markdown("### 📊 Comparaison des Trois Modèles")
            st.info("Analyse comparative des performances de tous les modèles")
            
            if st.button("🚀 Lancer la comparaison complète", key="compare_btn", type="primary"):
                with st.spinner("Chargement des modèles..."):
                    unet_model = load_unet_model(MODEL_CONFIG['unet']['path'], device)
                    yolo_model = load_yolo_model(MODEL_CONFIG['yolo']['path'])
                    hybrid_model = load_hybrid_model(MODEL_CONFIG['hybrid']['path'], device)
                
                # Vérifier quels modèles sont disponibles
                available_models = []
                if unet_model is not None:
                    available_models.append('unet')
                if yolo_model is not None:
                    available_models.append('yolo')
                if hybrid_model is not None:
                    available_models.append('hybrid')
                
                if len(available_models) == 0:
                    st.error("❌ Aucun modèle disponible. Veuillez uploader au moins un modèle.")
                else:
                    with st.spinner(f"Exécution de {len(available_models)} modèle(s) en cours..."):
                        results = {}
                        
                        # U-Net
                        if 'unet' in available_models:
                            seg_unet, pred_unet, time_unet, boxes_unet = predict_unet(image, unet_model, device)
                            results['unet'] = {
                                'image': seg_unet, 
                                'predictions': pred_unet, 
                                'time': time_unet,
                                'boxes': boxes_unet
                            }
                        
                        # YOLO
                        if 'yolo' in available_models:
                            seg_yolo, pred_yolo, time_yolo, boxes_yolo = predict_yolo(image, yolo_model)
                            results['yolo'] = {
                                'image': seg_yolo, 
                                'predictions': pred_yolo, 
                                'time': time_yolo,
                                'boxes': boxes_yolo
                            }
                        
                        # Hybride
                        if 'hybrid' in available_models:
                            seg_hybrid, pred_hybrid, time_hybrid, boxes_hybrid = predict_hybrid(image, hybrid_model, device)
                            if seg_hybrid is not None:
                                results['hybrid'] = {
                                    'image': seg_hybrid, 
                                    'predictions': pred_hybrid, 
                                    'time': time_hybrid,
                                    'boxes': boxes_hybrid
                                }
                    
                    st.success("✅ Comparaison terminée!")
                    
                    # Afficher les images en grille
                    st.markdown("### 🖼️ Résultats visuels")
                    
                    # Créer les colonnes selon la présence de ground truth
                    if ground_truth_boxes:
                        # 4 colonnes: GT + 3 modèles
                        num_cols = len(results) + 1
                        cols = st.columns(num_cols)
                        
                        # Première colonne: Ground Truth
                        with cols[0]:
                            st.markdown("**Ground Truth**")
                            image_with_gt = draw_ground_truth_boxes(image, ground_truth_boxes, CLASS_NAMES_YOLO)
                            st.image(image_with_gt, use_container_width=True)
                            st.caption(f"📦 {len(ground_truth_boxes)} dommages annotés")
                        
                        # Colonnes suivantes: Modèles
                        model_names = {'unet': 'U-Net Custom', 'yolo': 'YOLO Prédéfini', 'hybrid': 'Architecture Hybride'}
                        for idx, (model_key, result) in enumerate(results.items()):
                            with cols[idx + 1]:
                                st.markdown(f"**{model_names[model_key]}**")
                                st.image(result['image'], use_container_width=True)
                                st.metric("⏱️ Temps", f"{result['time']:.3f}s")
                    else:
                        # 3 colonnes: 3 modèles seulement
                        cols = st.columns(3)
                        
                        model_names = {'unet': 'U-Net Custom', 'yolo': 'YOLO Prédéfini', 'hybrid': 'Architecture Hybride'}
                        
                        for idx, (model_key, result) in enumerate(results.items()):
                            with cols[idx]:
                                st.markdown(f"**{model_names[model_key]}**")
                                st.image(result['image'], use_container_width=True)
                                st.metric("⏱️ Temps", f"{result['time']:.3f}s")
                    
                    # Graphiques de comparaison
                    st.markdown("### 📈 Analyse comparative")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_time = create_comparison_chart(results)
                        st.plotly_chart(fig_time, use_container_width=True)
                    
                    with col2:
                        fig_conf = create_confidence_chart(results)
                        if fig_conf:
                            st.plotly_chart(fig_conf, use_container_width=True)
                    
                    # Tableau récapitulatif
                    st.markdown("### 📋 Tableau récapitulatif")
                    
                    summary_data = []
                    for model_key, result in results.items():
                        row = {
                            'Modèle': model_names[model_key],
                            'Temps (s)': f"{result['time']:.3f}",
                            'Dommages détectés': len(result['predictions'])
                        }
                        
                        # Ajouter IoU et F1 si ground truth disponible
                        if ground_truth_boxes and result.get('boxes'):
                            metrics = calculate_metrics(result['boxes'], ground_truth_boxes)
                            row['IoU'] = f"{metrics['iou']:.3f}"
                            row['F1-Score'] = f"{metrics['f1']:.3f}"
                            row['Precision'] = f"{metrics['precision']:.3f}"
                            row['Recall'] = f"{metrics['recall']:.3f}"
                        else:
                            avg_conf = np.mean(list(result['predictions'].values())) if result['predictions'] else 0
                            row['Confiance moyenne'] = f"{avg_conf * 100:.1f}%"
                        
                        summary_data.append(row)
                    
                    import pandas as pd
                    df_summary = pd.DataFrame(summary_data)
                    st.dataframe(df_summary, use_container_width=True)
                    
                    # Recommandation
                    st.markdown("### 🏆 Recommandation")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    fastest = min(results.items(), key=lambda x: x[1]['time'])
                    most_detections = max(results.items(), key=lambda x: len(x[1]['predictions']))
                    
                    with col1:
                        st.success(f"**🚀 Plus rapide:** {model_names[fastest[0]]} ({fastest[1]['time']:.3f}s)")
                    with col2:
                        st.success(f"**🔍 Plus de détections:** {model_names[most_detections[0]]} ({len(most_detections[1]['predictions'])} types)")
                    
                    # Meilleur F1-Score si ground truth disponible
                    if ground_truth_boxes:
                        best_f1_model = None
                        best_f1_score = 0
                        
                        for model_key, result in results.items():
                            if result.get('boxes'):
                                metrics = calculate_metrics(result['boxes'], ground_truth_boxes)
                                if metrics['f1'] > best_f1_score:
                                    best_f1_score = metrics['f1']
                                    best_f1_model = model_key
                        
                        if best_f1_model:
                            with col3:
                                st.success(f"**🎯 Meilleur F1-Score:** {model_names[best_f1_model]} ({best_f1_score:.3f})")
    
    else:
        # Message d'accueil
        st.markdown("""
            <div style='text-align: center; padding: 50px;'>
                <h2>👋 Bienvenue dans le système de détection des dommages routiers</h2>
                <p style='font-size: 18px; color: #7f8c8d;'>
                    Chargez une image de route pour commencer l'analyse avec nos trois modèles de Deep Learning
                </p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()