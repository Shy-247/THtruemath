import pytesseract
from PIL import Image
import io
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sympy import *
from sympy import symbols, Eq, solve, latex
from sympy import symbols
from sympy import latex, nsimplify
import numpy as np
import cv2
from werkzeug.datastructures import FileStorage
import os
from pix2tex.cli import LatexOCR
import logging
from django.shortcuts import render
logger = logging.getLogger(__name__)
import re
from .preprocess_function import *
from .classification_function import *
from texify.inference import batch_inference
from texify.model.model import load_model
from texify.model.processor import load_processor
@csrf_exempt  # Nếu bạn không sử dụng CSRF token
def read_image(request):
    logger.debug(f"Request method: {request.method}")
    logger.debug(f"Files in request: {request.FILES}")

    if request.method == 'POST':
        if 'image' in request.FILES and 'cropped_image' in request.FILES and 'Ngu' in request.POST:
            try:
                #original_image = request.FILES['image']
                cropped_image = request.FILES['cropped_image']
                Ngu = request.POST.get('Ngu') #biến Ngu ở đây nhận 2 giá trị Cam hoặc Pdf tương ứng với 2 loại hình ảnh mà mình phân loại, mày muốn làm đéo gì thì làm if Ngu = A hay = B gì gì đó múa đi
                print(Ngu) #Ngu = pdf hoặc cam
                
                # Kiểm tra nội dung hình ảnh
                logger.debug(f"Cropped image name: {cropped_image.name}")

                #original_text = read_text_from_image(original_image, Ngu)
                cropped_text = read_symbol_from_image(cropped_image, Ngu)

                return JsonResponse({
                    'cropped_text': cropped_text
                })
            except Exception as e:
                logger.error(f"Error processing images: {str(e)}")
                return JsonResponse({'error': str(e)}, status=400)
        else:
            return JsonResponse({'error': 'Required files not provided'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)


def DefineTags(NhapVao):
    tags = set()
    NhapVao = NhapVao.lower()
    pt_count = NhapVao.count('phương trình')
    hpt_count = NhapVao.count('hệ phương trình')
    if hpt_count > 0 and pt_count == hpt_count:
        tags.add("hệ phương trình")
    if hpt_count > 0 and pt_count > hpt_count:
        tags.add('phương trình')
    if 'hàm số' in NhapVao:
        tags.add('hàm số')
    if 'rút gọn' in tags:
        tags.add('rút gọn')
    if 'có nghĩa' in tags:
        tags.add('có nghĩa')
    if 'trục căn thức' in tags:
        tags.add('trục căn thức')
    if 'điều kiện xác định' in tags:
        tags.add('điều kiện xác định')
    tags = ','.join(tags)
    return tags

def remove_formatting(latex_code):
    # Regex patterns to match various LaTeX formatting commands
    patterns = [
        r'\\textbf\{(.*?)\}',   # \textbf{...}
        r'\\textit\{(.*?)\}',   # \textit{...}
        r'\\textsf\{(.*?)\}',   # \textsf{...}
        r'\\texttt\{(.*?)\}',   # \texttt{...}
        r'\\textnormal\{(.*?)\}', # \textnormal{...}
        r'\\mathrm\{(.*?)\}',   # \mathrm{...}
        r'\\mathbf\{(.*?)\}',   # \mathbf{...}
        r'\\mathsf\{(.*?)\}',   # \mathsf{...}
        r'\\mathit\{(.*?)\}',   # \mathit{...}
        r'\\mathfrak\{(.*?)\}', # \mathfrak{...}
        r'\\mathcal\{(.*?)\}',  # \mathcal{...}
        r'\\bf\{(.*?)\}',       # \bf{...}
        r'\\it\{(.*?)\}',       # \it{...}
        r'\\sf\{(.*?)\}',       # \sf{...}
        r'\\tt\{(.*?)\}',       # \tt{...}
        r'\\normalfont\{(.*?)\}', # \normalfont{...}
        r'\\itshape\{(.*?)\}',  # \itshape{...}
        r'\\bfseries\{(.*?)\}', # \bfseries{...}
        r'\\itshape\{(.*?)\}'   # \itshape{...}
    ]
    for pattern in patterns:
        latex_code = re.sub(pattern, r'\1', latex_code)
    
    return latex_code
def remove_numeric_abs_values(latex_code):
    # Pattern to match |A| where A is numeric only (including negative numbers), with no alphabetic characters
    numeric_abs_pattern = re.compile(
        r'(?:\\bigg\||\\left\||\\right\||\\bigl\||\\bigr\||\||\\left\[|\\mid)'  # Opening absolute value symbol
        r'(-?\d+)'  # Numeric content, including optional negative sign
        r'(?:\\bigg\||\\left\||\\right\||\\bigl\||\\bigr\||\||\\right\]|\\mid)'  # Closing absolute value symbol
        , re.DOTALL
    )
    
    # Replace matches with an empty string
    cleaned_code = re.sub(numeric_abs_pattern, '', latex_code)
    
    return cleaned_code

def read_symbol_from_image(image_file, mode):
    try:
        image_data = image_file.read()
        image = Image.open(io.BytesIO(image_data))
        
        image_array = np.array(image)
        cv2.imwrite("temp/img.png", image_array)
    except Exception as e:
        print(f"Error loading or saving image: {e}")
        return None

    text_type = set()
    equ_type = set()

    # Segment the image into lines
    try:
        line_images = process_and_segment_image(image_array)
        if not line_images:
            print("No lines found in the image.")
            return None
    except Exception as e:
        print(f"Error processing images: {e}")
        return None

    model = load_model()
    processor = load_processor()
    custom_config = r'--oem 3 --psm 4 -l vie'

    # Loop through all lines once
    for i, line_img in enumerate(line_images):
        pil_image = Image.fromarray(cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY))

        # Skip very small or very large images
        if ((pil_image.size[0] < 35 and pil_image.size[1] < 35) or (pil_image.size[1] > 1000 and pil_image.size[0] > 1000)) and len(line_images) > 5:
            print(f"Line {i + 1}: Skipping OCR for image smaller than 35x35 or larger than 1000x1000")
            continue
        if mode == 'Cam':
            line_img = preprocess_otsuTH_image(resize(line_img))
            cv2.imwrite(f'temp/lines/line_{i+1}.png', line_img)
            pil_image = Image.fromarray(line_img)

        # Perform OCR on the line image to get text_type
        try:
            text = pytesseract.image_to_string(pil_image, config=custom_config)
        except Exception as e:
            print(f"Error during OCR: {e}")
            text = ""
        print(f'extracting image no_{i+1}')

        # Add extracted text_type to the set
        extracted_text_type = DefineTags(text)
        if 'phương trình' in extracted_text_type:
            break
        text_type.add(extracted_text_type)
        if 'rút gọn' in extracted_text_type or 'trục căn thức' in extracted_text_type or 'có nghĩa' in extracted_text_type or 'điều kiện xác định' in extracted_text_type:
            return 'căn thức'
        if 'hệ phương trình' in extracted_text_type:
            return 'hệ phương trình'

    # If no key text is found, continue to extract equ_type
    for i, line_img in enumerate(line_images):
        pil_image = Image.fromarray(cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY))

        if mode == 'Cam':
            line_img = preprocess_otsuTH_image(resize(line_img))
            cv2.imwrite(f'temp/lines/line_{i+1}.png', line_img)
            pil_image = Image.fromarray(line_img)

        # Perform LaTeX inference on the line image to get equ_type
        try:
            latex_code = batch_inference([pil_image], model, processor)
            latex_code = ''.join(latex_code)
            latex_code = remove_formatting(latex_code)
            latex_code = remove_numeric_abs_values(latex_code)
        except Exception as e:
            print(f"Error during LaTeX inference: {e}")
            latex_code = ""
        print(f'extracting content from image no_{i+1}')
        if r'begin{cases}' in latex_code:
            return 'hệ phương trình'

        # Add extracted equ_type to the set
        equ_type.add(EqClassifier(latex_code))

    # Return the combined equ_type
    return ','.join(equ_type)



def get_home(request): 
    return render(request, 'home.html')


def test_page(request):
    return render(request, 'test1.html')

def test_page2(request):
    return render(request, 'test2.html')

def test_page3(request):
    return render(request, 'test3.html')


def test_page4(request):
    return render(request, 'test4.html')

def test_page5(request):
    return render(request, 'test5.html')