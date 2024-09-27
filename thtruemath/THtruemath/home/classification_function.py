import re
import os
from pix2tex.cli import LatexOCR
import pytesseract
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2

#hàm loại bỏ một số định dạng matplotlib không hỗ trợ (dùng hỗ trợ cho hàm render_latex_to_png)
def preprocess_latex(latex_code):
    """
    Preprocesses LaTeX code to remove or replace unsupported commands.

    Args:
        latex_code (str): The LaTeX code to preprocess.

    Returns:
        str: The cleaned LaTeX code.
    """
    # Remove unsupported LaTeX commands like \scriptstyle
    cleaned_latex = re.sub(r'\\scriptstyle', '', latex_code)
    cleaned_latex = re.sub(r'\\textstyle', '', cleaned_latex)
    cleaned_latex = re.sub(r'\\displaystyle', '', cleaned_latex)
    # You can add more replacements if needed

    # Remove extra spaces (optional)
    cleaned_latex = re.sub(r'\s+', ' ', cleaned_latex).strip()

    return cleaned_latex

#hàm render code latex thành ảnh dùng để debug (không sử dụng trong dự án hoàn thiện)
def render_latex_to_png(latex_code):
    """
    Renders LaTeX code as a math expression to a PNG file.

    Args:
        latex_code (str): The LaTeX code to render.
    """
    import matplotlib.pyplot as plt
    import os

    # Define the output directory
    output_dir = r"temp"
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Use the LaTeX code as the filename
    output_path = os.path.join(output_dir, "ltc.png")

    # Preprocess the LaTeX code
    latex_code = preprocess_latex(latex_code)

    # Set up the plot with no axes
    plt.figure(figsize=(10, 2))
    plt.axis('off')
    try:
        # Use matplotlib to render the LaTeX code as math symbols
        plt.text(0.5, 0.5, f"${latex_code}$", fontsize=20, ha='center', va='center')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    except Exception as e:
        # If there's an error, render an image with "Invalid code"
        plt.clf()
        plt.text(0.5, 0.5, f"Error: {e}", fontsize=20, ha='center', va='center')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    finally:
        plt.close()

#hàm đổi dấu căn thành một kí tụ đặc biệt để tối ưu hiệu năng khi sử dụng regex
#Với việc này, thời gian chạy được giảm từ 2p còn dưới 5s
def preprocess_sqrt(latex):
    return latex.replace(r'\sqrt', '<')

#hàm phân loại các dạng phương trình vô tỉ (với kí hiệu dấu căn được đổi thành <{} thay vì \sqrt{})
def SqrtEqClassifier(latex_code):
    
    #phương trình dạng <{A} = B
    sqrt_0_pattern = re.compile(
        r'(^|(?<=\s))'  # Bắt đầu chuỗi hoặc đứng sau khoảng trắng
        r'(?:[^<]*)?'   # Bất kỳ ký tự nào không chứa '<'
        r'<\{'          # Ký hiệu mở '<{'
        r'(?=.*?[a-zA-Z])'
        r'([a-zA-Z0-9\\\|\(\)\{\}\[\]^\+\-\*/\s\.\,]+?)'  # Nội dung bên trong (A)
        r'\}'           # Đóng ngoặc
        r'(?:[^<]*)?'   # Bất kỳ ký tự nào không chứa '<'
        r'= *'          # Dấu bằng với khoảng trắng tùy chọn
        r'(?:[^<]*)?'   # Bất kỳ ký tự nào không chứa '<'
        r'([a-zA-Z0-9\\\(\)\{\}\[\]^\+\-\*/\s\.\,]+?)'  # Nội dung bên phải (B)
        r'(?:[^<]*)?'   # Bất kỳ ký tự nào không chứa '<'
        r'|'
        r'(^|(?<=\s))'  # Bắt đầu chuỗi hoặc đứng sau khoảng trắng
        r'([a-zA-Z0-9\\\(\)\{\}\[\]^\+\-\*/\s\.\,]+?)'  # Nội dung bên trái (B)
        r'(?:[^<]*)?'   # Bất kỳ ký tự nào không chứa '<'
        r'= *'          # Dấu bằng với khoảng trắng tùy chọn
        r'(?:[^<]*)?'   # Bất kỳ ký tự nào không chứa '<'
        r'<\{'          # Ký hiệu mở '<{'
        r'(?=.*?[a-zA-Z])'
        r'([a-zA-Z0-9\\\|\(\)\{\}\[\]^\+\-\*/\s\.\,]+?)'  # Nội dung bên trong (A)
        r'\}'           # Đóng ngoặc
        r'(?:[^<]*)?'   # Bất kỳ ký tự nào không chứa '<'
        , re.DOTALL
    )

    #phương trình dạng <{A} =<{B}
    sqrt_1_pattern = re.compile(
        r'(^|(?<=\s))'  # Bắt đầu chuỗi hoặc đứng sau khoảng trắng
        r'(?:[^<]*)?'   # Bất kỳ ký tự nào không chứa '<'
        r'<\{'          # Ký hiệu mở '<{'
        r'(?=.*?[a-zA-Z])'
        r'([a-zA-Z0-9\\\|\(\)\{\}\[\]^\+\-\*/\s\.\,]+?)'  # Nội dung bên trong (A)
        r'\}'           # Đóng ngoặc
        r'(?:[^<]*)?'   # Bất kỳ ký tự nào không chứa '<'
        r'= *'          # Dấu bằng với khoảng trắng tùy chọn
        r'(?:[^<]*)?'   # Bất kỳ ký tự nào không chứa '<'
        r'<\{'          # Ký hiệu mở '<{'
        r'(?=.*?[a-zA-Z])'
        r'([a-zA-Z0-9\\\|\(\)\{\}\[\]^\+\-\*/\s\.\,]+?)'  # Nội dung bên trong (B)
        r'\}'           # Đóng ngoặc
        r'(?:[^<]*)?'   # Bất kỳ ký tự nào không chứa '<'
        , re.DOTALL
    )

    #Phương trình dạng <{A} <{B} = C
    sqrt_2_pattern = re.compile(
        r'(^|(?<=\s))'  # Bắt đầu chuỗi hoặc đứng sau khoảng trắng
        r'(?:[^\<=]*?)'  # Bất kỳ ký tự nào ngoại trừ ký hiệu '<'
        r'([^=<]*?)'    # Bất kỳ ký tự nào không chứa '=' hoặc '<'
        r'= *'          # Dấu bằng với khoảng trắng tùy chọn
        r'[^=<]*?'      # Bất kỳ ký tự nào không chứa '=' hoặc '<'
        r'<\{'          # Ký hiệu mở '<{' cho A
        r'(?=.*?[a-zA-Z])' # Đảm bảo có ít nhất một chữ cái Latin trong A
        r'([a-zA-Z0-9\\\|\(\)\{\}\[\]^\+\-\*/\s\.\,]+)'   # Nội dung bên trong (A), không cho phép '}'
        r'\}'           # Đóng ngoặc cho A
        r'[^=<]*?'      # Bất kỳ ký tự nào không chứa '=' hoặc '<'
        r'<\{'          # Ký hiệu mở '<{' cho B
        r'(?=.*?[a-zA-Z])' # Đảm bảo có ít nhất một chữ cái Latin trong B
        r'([a-zA-Z0-9\\\|\(\)\{\}\[\]^\+\-\*/\s\.\,]+)'    # Nội dung bên trong (B), không cho phép '}'
        r'\}'           # Đóng ngoặc cho B
        r'[^=<]*?'      # Bất kỳ ký tự nào không chứa '=' hoặc '<'
        r'(?:[^\<=]*?)'  # Bất kỳ ký tự nào ngoại trừ ký hiệu '<'
        r'($|(?=\s))'   # Kết thúc chuỗi hoặc tiếp theo là khoảng trắng
        r'|'
        r'(^|(?<=\s))'  # Bắt đầu chuỗi hoặc đứng sau khoảng trắng
        r'(?:[^\<=]*?)'  # Bất kỳ ký tự nào ngoại trừ ký hiệu '<'
        r'<\{'          # Ký hiệu mở '<{' cho A
        r'(?=.*?[a-zA-Z])' # Đảm bảo có ít nhất một chữ cái Latin trong A
        r'([a-zA-Z0-9\\\|\(\)\{\}\[\]^\+\-\*/\s\.\,]+)'    # Nội dung bên trong (A), không cho phép '}'
        r'\}'           # Đóng ngoặc cho A
        r'[^=<]*?'      # Bất kỳ ký tự nào không chứa '=' hoặc '<'
        r'<\{'          # Ký hiệu mở '<{' cho B
        r'(?=.*?[a-zA-Z])' # Đảm bảo có ít nhất một chữ cái Latin trong B
        r'([a-zA-Z0-9\\\|\(\)\{\}\[\]^\+\-\*/\s\.\,]+)'    # Nội dung bên trong (B), không cho phép '}'
        r'\}'           # Đóng ngoặc cho B
        r'[^=<]*?'      # Bất kỳ ký tự nào không chứa '=' hoặc '<'
        r'= *'          # Dấu bằng với khoảng trắng tùy chọn
        r'(?:[^\<=]*?)'  # Bất kỳ ký tự nào ngoại trừ ký hiệu '<'
        r'($|(?=\s))'   # Kết thúc chuỗi hoặc tiếp theo là khoảng trắng
        , re.DOTALL
    )
    latex_code = preprocess_sqrt(latex_code)
    sqrt_1_match = sqrt_1_pattern.search(latex_code)
    sqrt_2_match = sqrt_2_pattern.search(latex_code)
    #abs_3_match = sqrt_3_pattern.search(latex_code)
    sqrt_0_match = sqrt_0_pattern.search(latex_code)
    if sqrt_2_match:
        return 'sqrt{A} + sqrt{B} = C'
    if sqrt_1_match:
        return 'sqrt{A} = sqrt{B}'
    if sqrt_0_match:
        return 'sqrt{A} = B'
    return 'Phương trình vô tỉ nâng cao'

#Hàm phân loại phương trình chứa dấu giá trị tuyệt đối
def AbsEqClassifier(latex_code):
    # Regular expressions for pattern matching
    # Pattern for |A| = B or B = |A|
    abs_0_pattern = re.compile(
        r'(^|(?<=\s))'  # Start of the string or preceded by space
        r'(?:[^\|=]*?)'  # Any characters except the absolute value symbol
        r'(?:\\bigg\||\\left\||\\right\||\\bigl\||\\bigr\||\||\\left\[|\\mid)'  # Opening absolute value symbol, includes \left[
        r'([a-zA-Z0-9\\\|\(\)\{\}\[\]^\+\-\*/\s\.\,]+)'  # Content inside absolute value symbols (A), allows LaTeX symbols and operators
        r'(?:\\bigg\||\\left\||\\right\||\\bigl\||\\bigr\||\||\\right\]|\\mid)'  # Closing absolute value symbol, includes \right]
        r'(?:[^\|=]*?)'  # Any characters except the absolute value symbol
        r'= *'  # Equal sign with optional spaces
        r'(?:[^\|=]*?)'  # Any characters except the absolute value symbol
        r'([a-zA-Z0-9\\\(\)\{\}\[\]^\+\-\*/\s\.\,]+)'  # Match content on the right side (B), allows LaTeX symbols and operators, excludes the absolute value symbol
        r'(?:[^\|=]*?)'  # Any characters except the absolute value symbol
        r'|'
        r'(^|(?<=\s))'  # Start of the string or preceded by space
        r'(?:[^\|=]*?)'  # Any characters except the absolute value symbol
        r'([a-zA-Z0-9\\\(\)\{\}\[\]^\+\-\*/\s\.\,]+)'  # Match content on the left side (B), allows LaTeX symbols and operators, excludes the absolute value symbol
        r'(?:[^\|=]*?)'  # Any characters except the absolute value symbol
        r'= *'  # Equal sign with optional spaces
        r'(?:[^\|=]*?)'  # Any characters except the absolute value symbol
        r'(?:\\bigg\||\\left\||\\right\||\\bigl\||\\bigr\||\||\\left\[|\\mid)'  # Opening absolute value symbol, includes \left[
        r'([a-zA-Z0-9\\\|\(\)\{\}\[\]^\+\-\*/\s\.\,]+)'  # Content inside absolute value symbols (A), allows LaTeX symbols and operators
        r'(?:\\bigg\||\\left\||\\right\||\\bigl\||\\bigr\||\||\\right\]|\\mid)'  # Closing absolute value symbol, includes \right]
        r'(?:[^\|=]*?)',  # Any characters except the absolute value symbol
        re.DOTALL
    )

    # Pattern for |A| = |B|
    abs_1_pattern = re.compile(
        r'(^|(?<=\s))'  # Start of the string or preceded by space
        r'(?:[^\|=]*?)'  # Any characters except the absolute value symbol
        r'(?:\\bigg\||\\left\||\\right\||\\bigl\||\\bigr\||\||\\left\[|\\mid)'  # Opening absolute value symbol, includes \left[
        r'([a-zA-Z0-9\\\|\(\)\{\}\[\]^\+\-\*/\s\.\,]+)'  # Content inside absolute value symbols (A), allows LaTeX symbols and operators
        r'(?:\\bigg\||\\left\||\\right\||\\bigl\||\\bigr\||\||\\right\]|\\mid)'  # Closing absolute value symbol, includes \right]
        r'(?:[^\|=]*?)'  # Any characters except the absolute value symbol
        r'= *'  # Equal sign with optional spaces
        r'(?:[^\|=]*?)'  # Any characters except the absolute value symbol
        r'(?:\\bigg\||\\left\||\\right\||\\bigl\||\\bigr\||\||\\left\[|\\mid)'  # Opening absolute value symbol, includes \left[
        r'([a-zA-Z0-9\\\|\(\)\{\}\[\]^\+\-\*/\s\.\,]+)'  # Content inside absolute value symbols (A), allows LaTeX symbols and operators
        r'(?:\\bigg\||\\left\||\\right\||\\bigl\||\\bigr\||\||\\right\]|\\mid)'  # Closing absolute value symbol, includes \right]
        r'(?:[^\|=]*?)',  # Any characters except the absolute value symbol
        re.DOTALL
    )
    #Pattern for |A| + |B| = C
    abs_2_pattern = re.compile(
        r'(^|(?<=\s))'  # Start of the string or preceded by space
        r'(?:[^\|=]*?)'  # Any characters except the absolute value or equal sign
        r'(?:\\bigg\||\\left\||\\right\||\\bigl\||\\bigr\||\||\\left\[|\\mid)'  # Opening absolute value symbol, includes \left[
        r'([a-zA-Z0-9\\\|\(\)\{\}\[\]^\+\-\*/\s\.\,]+)'  # Content inside absolute value symbols (A), allows LaTeX symbols and operators
        r'(?:\\bigg\||\\left\||\\right\||\\bigl\||\\bigr\||\||\\right\]|\\mid)'  # Closing absolute value symbol, includes \right]
        r'(?:[^\|=]*?)'  # Any characters except the absolute value or equal sign
        r'(?:\\bigg\||\\left\||\\right\||\\bigl\||\\bigr\||\||\\left\[|\\mid)'  # Opening absolute value symbol, includes \left[
        r'([a-zA-Z0-9\\\|\(\)\{\}\[\]^\+\-\*/\s\.\,]+)'  # Content inside absolute value symbols (B), allows LaTeX symbols and operators
        r'(?:\\bigg\||\\left\||\\right\||\\bigl\||\\bigr\||\||\\right\]|\\mid)'  # Closing absolute value symbol, includes \right]
        r'(?:[^\|=]*?)'  # Any characters except the absolute value or equal sign
        r'= *'  # Equal sign with optional spaces
        r'(?:[^\|=]*?)'  # Any characters except the absolute value or equal sign
        r'([a-zA-Z0-9\\\(\)\{\}\[\]^\+\-\*/\s\.\,]+)'  # Match content on the right side (C), allows LaTeX symbols and operators, excludes the absolute value symbol
        r'(?:[^\|=]*?)'  # Any characters except the absolute value or equal sign
        r'|'
        r'(^|(?<=\s))'  # Start of the string or preceded by space
        r'(?:[^\|=]*?)'  # Any characters except the absolute value or equal sign
        r'([a-zA-Z0-9\\\(\)\{\}\[\]^\+\-\*/\s\.\,]+)'  # Match content on the left side (C), allows LaTeX symbols and operators, excludes the absolute value symbol
        r'(?:[^\|=]*?)'  # Any characters except the absolute value or equal sign
        r'= *'  # Equal sign with optional spaces
        r'(?:[^\|=]*?)'  # Any characters except the absolute value or equal sign
        r'(?:\\bigg\||\\left\||\\right\||\\bigl\||\\bigr\||\||\\left\[|\\mid)'  # Opening absolute value symbol, includes \left[
        r'([a-zA-Z0-9\\\|\(\)\{\}\[\]^\+\-\*/\s\.\,]+)'  # Content inside absolute value symbols (A), allows LaTeX symbols and operators
        r'(?:\\bigg\||\\left\||\\right\||\\bigl\||\\bigr\||\||\\right\]|\\mid)'  # Closing absolute value symbol, includes \right]
        r'(?:[^\|=]*?)'  # Any characters except the absolute value or equal sign
        r'(?:\\bigg\||\\left\||\\right\||\\bigl\||\\bigr\||\||\\left\[|\\mid)'  # Opening absolute value symbol, includes \left[
        r'([a-zA-Z0-9\\\|\(\)\{\}\[\]^\+\-\*/\s\.\,]+)'  # Content inside absolute value symbols (B), allows LaTeX symbols and operators
        r'(?:\\bigg\||\\left\||\\right\||\\bigl\||\\bigr\||\||\\right\]|\\mid)'  # Closing absolute value symbol, includes \right]
        r'(?:[^\|=]*?)',  # Any characters except the absolute value or equal sign
        re.DOTALL
    )
    # Pattern to match \frac{A}{|A'|} + \frac{B}{|B'|} = C
    abs_3_pattern = re.compile(
        r'(^|(?<=\s))'  # Start of the string or preceded by space
        r'(?:[^\|=]*?)'  # Any characters except the absolute value symbol
        r'\\frac'  # Match the \frac command
        r'\{'  # Opening brace for the numerator
        r'([a-zA-Z0-9\\\|\(\)\{\}\[\]^\+\-\*/\s\.\,]+)'  # Content of the numerator A
        r'\}'  # Closing brace for the numerator
        r'\{'  # Opening brace for the denominator
        r'(?:[^\|=]*?)'  # Any characters except the absolute value symbol
        r'(?:\\bigg\||\\left\||\\right\||\\bigl\||\\bigr\||\||\\left\[|\\mid)'  # Opening absolute value symbol
        r'([a-zA-Z0-9\\\|\(\)\{\}\[\]^\+\-\*/\s\.\,]+)'  # Content inside absolute value symbols (A'), allows LaTeX symbols and operators
        r'(?:\\bigg\||\\left\||\\right\||\\bigl\||\\bigr\||\||\\right\]|\\mid)'  # Closing absolute value symbol
        r'\}'  # Closing brace for the denominator
        r'(?:[^\|=]*?)'  # Any characters except the absolute value symbol
        r'(?:\\frac'  # Optional second \frac command
        r'\{'  # Opening brace for the second numerator
        r'([a-zA-Z0-9\\\|\(\)\{\}\[\]^\+\-\*/\s\.\,]+)'  # Content of the second numerator B
        r'\}'  # Closing brace for the second numerator
        r'\{'  # Opening brace for the second denominator
        r'(?:[^\|=]*?)'  # Any characters except the absolute value symbol
        r'(?:\\bigg\||\\left\||\\right\||\\bigl\||\\bigr\||\||\\left\[|\\mid)'  # Opening absolute value symbol for second fraction
        r'([a-zA-Z0-9\\\|\(\)\{\}\[\]^\+\-\*/\s\.\,]+)'  # Content inside absolute value symbols (B'), allows LaTeX symbols and operators
        r'(?:\\bigg\||\\left\||\\right\||\\bigl\||\\bigr\||\||\\right\]|\\mid)'  # Closing absolute value symbol for second fraction
        r'\}'  # Closing brace for the second denominator
        r'(?:[^\|]*?))?'  # Any characters except the absolute value symbol; make this entire section optional
        r'= *'  # Equal sign with optional spaces
        r'(?:[^\|=]*?)'  # Any characters except the absolute value symbol
        r'([a-zA-Z0-9\\\(\)\{\}\[\]^\+\-\*/\s\.\,]+)'  # Content for C
        r'(?:[^\|=]*?)',  # Any characters except the absolute value symbol
        re.DOTALL
    )

    # Match patterns
    abs_1_match = abs_1_pattern.search(latex_code)
    abs_2_match = abs_2_pattern.search(latex_code)
    abs_3_match = abs_3_pattern.search(latex_code)
    abs_0_match = abs_0_pattern.search(latex_code)

    # Check for ABS_2 first, as it is more specific
        # Check for ABS_3
    if abs_3_match:
        return 'Giá trị tuyệt đối ở mẫu'
    
    if abs_2_match:
        return '|A| + |B| = C'
    
    # Check for ABS_1
    if abs_1_match:
        return '|A| = |B|'




    
    # Check for ABS_0 last, as it is more general
    if abs_0_match:
        return '|A| = B'

    # Default case for advanced absolute value equations
    return 'Phương trình chứa dấu giá trị tuyệt đối nâng cao'
def EqClassifier(latex_code):
    latex_code = latex_code.lower()
    tags = []
    # Xác định xem trong ảnh có chứa ẩn x không
    if 'x' not in latex_code and r'\chi' not in latex_code and r'\alpha' not in latex_code:
        tags.append('phân loại phương trình không thành công')
    else:
        # Xác định xem trong ảnh có chứa dấu căn không
        # Define the regular expression pattern to match \sqrt{ followed by x
        pattern_1 =r'\\sqrt\{[^}]*[a-zA-Z][^}]*\}'
        # Define the updated regular expression pattern to match any letter except x
        pattern_2 = r'\\sqrt\[\d+\]\{[^}]*[a-zA-Z][^}]*\}'
        # Check for the pattern in the LaTeX code
        if re.search(pattern_1, latex_code) or re.search(pattern_2, latex_code):
            print('phát hiện phương trình vô tỉ')
            tags.append(SqrtEqClassifier(latex_code))

        # Xác định xem trong ảnh có chứa kí hiệu giá trị tuyệt đối không
        elif any(abs_symbol in latex_code for abs_symbol in ('|', '\bigg|', '\left|', '\right|', '\bigl|', '\bigr|', '\left[', '\right]')):
            tags.append(AbsEqClassifier(latex_code))
        
        
        # Check for the pattern in the LaTeX code
        elif re.search('m', latex_code):
            tags.append('Phương trình tham số')
                
        elif 'x^{2' in latex_code or 'x}^{2' in latex_code:
            if 'x^{4}' in latex_code or 'x}^{4}' in latex_code:
                tags.append('Phương trình trùng phương')
            else:
                tags.append('Phương trình bậc hai cơ bản')
        
        if len(tags) == 0:
            tags.append('Phương trình bậc nhất')

    return ','.join(tags)
