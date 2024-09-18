import re
import os
from pix2tex.cli import LatexOCR
import pytesseract
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
def read_text_from_image(image): 
    custom_config = r'--oem 3 --psm 4 -l vie -c tessedit_do_invert=0 -c textord_tabfind_find_rows=0'
    recognized_text = pytesseract.image_to_string(image, config = custom_config)
    return recognized_text
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
def preprocess_sqrt(latex):
    return latex.replace(r'\sqrt{', '<{')
def SqrtEqClassifier(latex_code):
    sqrt_0_pattern = re.compile(
        r'(^|(?<=\s))'  # Start of the string or preceded by space
        r'(?:[^<]*)?'   # Any characters not containing '<'
        r'<\{'          # Opening '<{' symbol
        r'([a-zA-Z0-9\\\|\(\)\{\}\[\]^\+\-\*/\s\.\,\<]+?)'  # Content inside (A)
        r'\}'           # Closing brace
        r'(?:[^<]*)?'   # Any characters not containing '<'
        r'= *'          # Equal sign with optional spaces
        r'(?:[^<]*)?'   # Any characters not containing '<'
        r'([a-zA-Z0-9\\\(\)\{\}\[\]^\+\-\*/\s\.\,]+?)'  # Content on the right side (B)
        r'(?:[^<]*)?'   # Any characters not containing '<'
        r'|'
        r'(^|(?<=\s))'  # Start of the string or preceded by space
        r'([a-zA-Z0-9\\\(\)\{\}\[\]^\+\-\*/\s\.\,]+?)'  # Content on the left side (B)
        r'(?:[^<]*)?'   # Any characters not containing '<'
        r'= *'          # Equal sign with optional spaces
        r'(?:[^<]*)?'   # Any characters not containing '<'
        r'<\{'          # Opening '<{' symbol
        r'([a-zA-Z0-9\\\|\(\)\{\}\[\]^\+\-\*/\s\.\,\<]+?)'  # Content inside (A)
        r'\}'           # Closing brace
        r'(?:[^<]*)?'   # Any characters not containing '<'
        , re.DOTALL
    )

    # Pattern for <{A} = <{B}
    sqrt_1_pattern = re.compile(
        r'(^|(?<=\s))'  # Start of the string or preceded by space
        r'(?:[^<]*)?'   # Any characters not containing '<'
        r'<\{'          # Opening '<{' symbol
        r'([a-zA-Z0-9\\\|\(\)\{\}\[\]^\+\-\*/\s\.\,]+?)'  # Content inside (A)
        r'\}'           # Closing brace
        r'(?:[^<]*)?'   # Any characters not containing '<'
        r'= *'          # Equal sign with optional spaces
        r'(?:[^<]*)?'   # Any characters not containing '<'
        r'<\{'          # Opening '<{' symbol
        r'([a-zA-Z0-9\\\|\(\)\{\}\[\]^\+\-\*/\s\.\,]+?)'  # Content inside (B)
        r'\}'           # Closing brace
        r'(?:[^<]*)?'   # Any characters not containing '<'
        , re.DOTALL
    )

    # Pattern for <{A} <{B} = C
    sqrt_2_pattern = re.compile(
        r'(^|(?<=\s))'  # Start of the string or preceded by space
        r'[^=<]*?'      # Any characters not containing '=' or '<'
        r'<\{'          # Opening '<{' symbol for A
        r'([a-zA-Z0-9\\\|\(\)\{\}\[\]^\+\-\*/\s\.\,\<\;]+?)'  # Content inside (A), no '<' allowed
        r'\}'           # Closing brace
        r'[^=<]*?'      # Any characters not containing '=' or '<'
        r'<\{'          # Opening '<{' symbol for B
        r'([a-zA-Z0-9\\\|\(\)\{\}\[\]^\+\-\*/\s\.\,\<\;]+?)'  # Content inside (B), no '<' allowed
        r'\}'           # Closing brace
        r'[^=<]*?'      # Any characters not containing '=' or '<'
        r'= *'          # Equal sign with optional spaces
        r'([^<]+?)'     # Content on the right side (C), ensures no '<'
        r'($|(?=\s))',  # End of the string or followed by space
        re.DOTALL
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
    tags = []
    # Xác định xem trong ảnh có chứa ẩn x không
    if 'x' not in latex_code and r'\chi' not in latex_code and r'\alpha' not in latex_code:
        tags.append('phân loại phương trình không thành công')
    else:
        # Xác định xem trong ảnh có chứa dấu căn không
        # Define the regular expression pattern to match \sqrt{ followed by x
        pattern =r'\\sqrt\{[^}]*[a-zA-Z][^}]*\}'
        # Define the updated regular expression pattern to match any letter except x
        combined_pattern = r'(\{(?!x)[a-zA-Z]\}|(?!x)[a-zA-Z])\s*[\+\-]\s*\d+|[\+\-]\s*\d*\s*(\{(?!x)[a-zA-Z]\}|(?!x)[a-zA-Z])'

        # Check for the pattern in the LaTeX code
        if re.search(pattern, latex_code):
            print('phát hiện phương trình vô tỉ')
            tags.append(SqrtEqClassifier(latex_code))

        # Xác định xem trong ảnh có chứa kí hiệu giá trị tuyệt đối không
        elif any(abs_symbol in latex_code for abs_symbol in ('|', '\bigg|', '\left|', '\right|', '\bigl|', '\bigr|', '\left[', '\right]')):
            tags.append(AbsEqClassifier(latex_code))
        
        
        # Check for the pattern in the LaTeX code
        elif re.search(combined_pattern, latex_code):
            tags.append('Phương trình tham số')
                
        elif 'x^{2}' in latex_code or 'mathbf{x}^{2}' in latex_code:
            if '\sqrt{' not in latex_code and not any(abs_symbol in latex_code for abs_symbol in ('|', '\bigg|', '\left|')):
                if 'x^{4}' in latex_code or 'mathbf{x}^{4}' in latex_code:
                    tags.append('Phương trình trùng phương')
                else:
                    tags.append('Phương trình bậc hai cơ bản')
        
        if len(tags) == 0:
            tags.append('Phương trình bậc nhất')

    return ','.join(tags)
latex_code = r'\sqrt{2x}x = 3'
print(EqClassifier(latex_code))