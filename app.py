import cv2 #image and video processing
import torch #deep learning and tensor computation
from PIL import Image #image processing
from transformers import BlipProcessor, BlipForConditionalGeneration #BLIP model for image captioning

# Load BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Caption generation function
def generate_caption(frame):
    image = Image.fromarray(frame)
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Start camera
cap = cv2.VideoCapture(0)
frame_count = 0
caption = ""

print("Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for captioning model input (optional)
    resized_frame = cv2.resize(frame, (384, 384))

    # Generate caption every 30 frames (~1 sec if 30 FPS)
    if frame_count % 30 == 0:
        try:
            caption = generate_caption(resized_frame)
            print("Caption:", caption)
        except Exception as e:
            print("Captioning error:", e)
            caption = ""

    frame_count += 1

    # Show caption on the video
    cv2.putText(frame, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Real-Time AI Scene Description", frame)

    # Quit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
