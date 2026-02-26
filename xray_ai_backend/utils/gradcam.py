import os
import cv2
import torch
import numpy as np
import traceback

def get_target_layer(model):
    for name in ["denseblock4", "denseblock3"]:
        try:
            layer = getattr(model.features, name)
            print(f"[GradCAM] Target layer: model.features.{name}")
            return layer
        except AttributeError:
            pass

    for name in ["layer4", "layer3"]:
        try:
            layer = getattr(model, name)
            print(f"[GradCAM] Target layer: model.{name}")
            return layer
        except AttributeError:
            pass

    try:
        layer = model.features[-1]
        print("[GradCAM] Target layer: model.features[-1]")
        return layer
    except (AttributeError, TypeError, IndexError):
        pass

    last_conv, last_name = None, None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv, last_name = module, name

    if last_conv is not None:
        print(f"[GradCAM] Fallback — last Conv2d: {last_name}")
        return last_conv

    raise RuntimeError("[GradCAM] No suitable target layer found.")

def get_lung_mask(gray_image, h, w):
    try:
        img_8u = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_eq = clahe.apply(img_8u)

        _, thresh = cv2.threshold(img_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  np.ones((10, 10), np.uint8))

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        lung_mask = np.zeros((h, w), dtype=np.float32)

        if num_labels > 1:
            areas         = stats[1:, cv2.CC_STAT_AREA]
            sorted_labels = np.argsort(areas)[::-1] + 1
            for i in range(min(2, len(sorted_labels))):
                lung_mask[labels == sorted_labels[i]] = 1.0

        lung_mask = cv2.dilate(lung_mask, np.ones((25, 25), np.uint8))
        lung_mask = cv2.GaussianBlur(lung_mask, (31, 31), 0)
        lung_mask = np.clip(lung_mask, 0, 1)

        if lung_mask.sum() < (h * w * 0.05):
            print("[GradCAM] Lung mask fallback: center crop")
            lung_mask = np.zeros((h, w), dtype=np.float32)
            lung_mask[int(h * 0.08):int(h * 0.92), int(w * 0.08):int(w * 0.92)] = 1.0

        return lung_mask

    except Exception as e:
        print(f"[GradCAM] Lung mask error: {e} — using full image")
        return np.ones((h, w), dtype=np.float32)

def merge_boxes(boxes, merge_threshold=0.25, image_w=1, image_h=1):
    if not boxes:
        return []

    boxes    = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    merge_dx = image_w * merge_threshold
    merge_dy = image_h * merge_threshold
    merged   = []
    used     = [False] * len(boxes)

    for i, (x1, y1, w1, h1) in enumerate(boxes):
        if used[i]:
            continue
        cx1   = x1 + w1 / 2
        cy1   = y1 + h1 / 2
        group = [(x1, y1, x1 + w1, y1 + h1)]
        used[i] = True

        for j, (x2, y2, w2, h2) in enumerate(boxes):
            if used[j]:
                continue
            if abs(cx1 - (x2 + w2/2)) < merge_dx and abs(cy1 - (y2 + h2/2)) < merge_dy:
                group.append((x2, y2, x2 + w2, y2 + h2))
                used[j] = True

        gx1 = min(b[0] for b in group)
        gy1 = min(b[1] for b in group)
        gx2 = max(b[2] for b in group)
        gy2 = max(b[3] for b in group)
        merged.append((gx1, gy1, gx2 - gx1, gy2 - gy1))

    return merged

def _compute_cam(model, img_tensor, class_index, target_layer):
    activations, gradients, handles = [], [], []

    try:
        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                gradients.append(grad_output[0].detach().clone())

        handles.append(target_layer.register_forward_hook(forward_hook))
        handles.append(target_layer.register_full_backward_hook(backward_hook))

        output = model(img_tensor)

        if class_index >= output.shape[1]:
            print(f"[GradCAM] class_index {class_index} out of range ({output.shape[1]})")
            return None

        model.zero_grad()
        output[0, class_index].backward(retain_graph=True)

        if not activations or not gradients:
            print(f"[GradCAM] No activations/gradients for class {class_index}")
            return None

        activation = activations[0].detach()
        gradient   = gradients[0]
        weights     = gradient.mean(dim=(2, 3), keepdim=True)
        pos_weights = torch.clamp(weights, min=0)
        cam         = (pos_weights * activation).sum(dim=1).squeeze().cpu().numpy()

        if cam.max() <= 1e-6:
            cam = (torch.abs(weights) * activation).sum(dim=1).squeeze().cpu().numpy()

        if cam.max() <= 1e-6:
            print(f"[GradCAM] No CAM signal for class {class_index}")
            return None

        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    finally:
        for h in handles:
            h.remove()

DISEASE_COLORS = [
    (255, 220,   0),   
    (0,   200, 255),  
    (255, 100,   0),  
    (180,   0, 255),   
    (0,   255, 120),   
]

def _draw_box_with_label(overlay, x1, y1, x2, y2, label, color, w, h):
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, w / 900)
    thickness  = max(2, w // 450)

    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), 5)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color,     3)

    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    label_y = y1 - 10
    if label_y - text_h < 0:
        label_y = y1 + text_h + 6

    bg_x1 = max(0,     x1)
    bg_y1 = max(0,     label_y - text_h - baseline)
    bg_x2 = min(w - 1, x1 + text_w + 8)
    bg_y2 = min(h - 1, label_y + baseline)

    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.putText(overlay, label, (x1 + 3, label_y),
                font, font_scale, color, thickness, cv2.LINE_AA)
    
def generate_gradcam(
    model,
    image_tensor,
    image_path,
    filename,
    class_index,
    disease_name=None,
    target_layer=None,
    alpha=0.50,
    output_dir="reports"
):
    
    diseases = [(disease_name or "Finding", class_index)]
    return generate_gradcam_multi(
        model=model,
        image_tensor=image_tensor,
        image_path=image_path,
        filename=filename,
        diseases=diseases,
        target_layer=target_layer,
        alpha=alpha,
        output_dir=output_dir
    )

def generate_gradcam_multi(
    model,
    image_tensor,
    image_path,
    filename,
    diseases,                
    target_layer=None,
    alpha=0.50,
    output_dir="reports"
):
    """
    Generates one GradCAM heatmap image with ALL detected diseases annotated.
    Each disease gets:
      - Its own CAM computed via a separate backward pass
      - Its own distinctly colored bounding box
      - Its own label drawn above its box

    Args:
        model        : Trained PyTorch model
        image_tensor : Preprocessed tensor [1, C, H, W] — outside torch.no_grad()
        image_path   : Full path to original image
        filename     : Output filename
        diseases     : List of (disease_name, class_index) e.g.
                       [("Atelectasis", 0), ("Consolidation", 5)]
        target_layer : Optional specific layer — auto-detected if None
        alpha        : Heatmap blend strength (default 0.50)
        output_dir   : Output folder (default "reports")

    Returns:
        str : Path to saved heatmap image, or None on failure
    """
    try:
        if not os.path.exists(image_path):
            print(f"[GradCAM] ERROR: Image not found: {image_path}")
            return None
        if image_tensor is None:
            print("[GradCAM] ERROR: image_tensor is None")
            return None
        if not diseases:
            print("[GradCAM] ERROR: No diseases provided")
            return None

        model.eval()
        device = next(model.parameters()).device

        if target_layer is None:
            target_layer = get_target_layer(model)

        original_bgr = cv2.imread(image_path)
        if original_bgr is None:
            print(f"[GradCAM] ERROR: cv2.imread failed: {image_path}")
            return None

        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        h, w, _      = original_rgb.shape
        gray      = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
        lung_mask = get_lung_mask(gray, h, w)
        overlay = original_rgb.copy().astype(np.float32)

        all_boxes = []  

        for i, (disease_name, class_index) in enumerate(diseases):
            color = DISEASE_COLORS[i % len(DISEASE_COLORS)]
            print(f"[GradCAM] Processing: {disease_name} (class {class_index})")

            img_tensor = image_tensor.clone().detach().to(device)
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            img_tensor.requires_grad_(True)

            cam = _compute_cam(model, img_tensor, class_index, target_layer)

            if cam is None:
                print(f"[GradCAM] Skipping {disease_name} — no CAM signal")
                continue

            cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
            cam_resized = cv2.GaussianBlur(cam_resized, (35, 35), 0)

            cam_masked = cam_resized * lung_mask
            if cam_masked.max() > 1e-6:
                cam_masked = (cam_masked - cam_masked.min()) / (cam_masked.max() - cam_masked.min() + 1e-8)
            else:
                cam_masked = cam_resized

            cam_uint8   = (cam_masked * 255).astype(np.uint8)
            heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

            cam_weight  = cam_masked[..., np.newaxis]
            overlay     = (
                overlay * (1 - alpha * cam_weight) +
                heatmap_rgb * (alpha * cam_weight)
            )

            active_vals = cam_masked[cam_masked > 0.1]
            if len(active_vals) == 0:
                continue

            hot_thresh = np.percentile(active_vals, 80)
            hot_region = (cam_masked >= hot_thresh).astype(np.uint8)
            hot_region = cv2.morphologyEx(hot_region, cv2.MORPH_CLOSE, np.ones((61, 61), np.uint8))
            hot_region = cv2.morphologyEx(hot_region, cv2.MORPH_OPEN,  np.ones((21, 21), np.uint8))

            contours, _ = cv2.findContours(hot_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            min_area  = h * w * 0.02
            raw_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]
            merged    = merge_boxes(raw_boxes, merge_threshold=0.25, image_w=w, image_h=h)
            
            if merged:
                bx, by, bw, bh = max(merged, key=lambda b: b[2] * b[3])
                pad = 10
                x1  = max(0,     bx - pad)
                y1  = max(0,     by - pad)
                x2  = min(w - 1, bx + bw + pad)
                y2  = min(h - 1, by + bh + pad)
                all_boxes.append((x1, y1, x2, y2, disease_name, color))

        if not all_boxes:
            print("[GradCAM] No boxes to draw — returning None")
            return None

        overlay = overlay.clip(0, 255).astype(np.uint8)

        for (x1, y1, x2, y2, label, color) in all_boxes:
            _draw_box_with_label(overlay, x1, y1, x2, y2, label, color, w, h)

        print(f"[GradCAM] Total boxes drawn: {len(all_boxes)} → {[b[4] for b in all_boxes]}")

        os.makedirs(output_dir, exist_ok=True)
        safe_name    = os.path.basename(filename)
        heatmap_path = os.path.join(output_dir, f"heatmap_{safe_name}")

        success = cv2.imwrite(heatmap_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if not success:
            print(f"[GradCAM] ERROR: cv2.imwrite failed: {heatmap_path}")
            return None

        print(f"[GradCAM] Saved → {heatmap_path}")
        return heatmap_path

    except Exception as e:
        print(f"[GradCAM] EXCEPTION: {e}")
        traceback.print_exc()
        return None