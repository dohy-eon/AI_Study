from flask import Flask, render_template, request, jsonify
from diffusers import SanaPipeline
from diffusers.utils import load_image
import torch
from PIL import Image
import os
import traceback

app = Flask(__name__)

# 업로드된 이미지와 생성된 이미지를 저장할 폴더
UPLOAD_FOLDER = 'static/uploads'
GENERATED_FOLDER = 'static/generated'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)

# 모델 파이프라인 변수 초기화
pipeline = None

try:
    # 1. SANA 모델 경로 지정 (문서에 명시된 모델)
    sana_model_path = "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"
    
    # 2. SanaPipeline 클래스로 모델 로드
    # 문서에 따라 torch.bfloat16 데이터 타입 사용
    pipeline = SanaPipeline.from_pretrained(
        sana_model_path,
        torch_dtype=torch.bfloat16
    )

    # 3. 모델을 GPU(CUDA)로 이동
    pipeline.to("cuda")

    # 4. 개별 컴포넌트의 데이터 타입 설정 (문서에 명시된 필수 단계)
    # Sana-1.5 모델은 vae와 text_encoder를 bfloat16으로 설정해야 합니다.
    pipeline.vae.to(torch.bfloat16)
    pipeline.text_encoder.to(torch.bfloat16)

    print("SANA 모델이 SanaPipeline으로 로드되었습니다! 이제 이미지 생성이 가능합니다.")

except Exception as e:
    print(f"모델 로드 실패: {e}")
    traceback.print_exc()
    pipeline = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    if not pipeline:
        return jsonify({'error': 'AI 모델이 로드되지 않았습니다. 서버 로그를 확인하세요.'}), 500
        
    prompt = request.form.get('prompt', 'a beautiful landscape')
    style_choice = request.form.get('style', 'monet')

    # **SANA 모델을 사용하여 텍스트 기반으로 이미지 생성**
    # SANA 모델은 기본적으로 텍스트-이미지 모델이므로, `image` 인자는 필요하지 않을 수 있습니다.
    # 만약 SANA-ControlNet을 사용하고 싶다면, 해당 파이프라인을 사용해야 합니다.
    # 하지만 현재 코드는 텍스트 기반 생성만 지원합니다.
    generated_image = pipeline(
        prompt=f"{prompt}, {style_choice} style", 
        height=512,  # 모델에 맞는 높이와 너비 지정
        width=512,
        guidance_scale=4.5,
        num_inference_steps=20,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).images[0]
    
    # 생성된 이미지를 static/generated 폴더에 저장
    # 파일명은 시간으로 구분하여 중복 방지
    import uuid
    generated_filename = f"generated_{str(uuid.uuid4())}.png"
    generated_path = os.path.join(GENERATED_FOLDER, generated_filename)
    generated_image.save(generated_path)

    # 생성된 이미지의 URL을 클라이언트에 반환
    return jsonify({'image_url': f'/static/generated/{generated_filename}'})

if __name__ == '__main__':
    app.run(debug=True)