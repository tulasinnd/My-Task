import cv2
import numpy as np

def get_output_image(original_image_path: str, fully_annotated_image_path: str, partially_annotated_image_path: str):
    # -------------------------LOAD FULLY ANNOTATED AND ORIGINAL IMAGES------------------------- #
    fully_annotated_image = cv2.imread(original_image_path)
    original_image = cv2.imread(fully_annotated_image_path)
    original_image = cv2.resize(original_image, (fully_annotated_image.shape[1], fully_annotated_image.shape[0])) 

    # ---------------------PRE-PROCESS THE IMAGE------------------------------- #
    # BITWISE OR
    image = cv2.bitwise_or(original_image, fully_annotated_image)
    #image = cv2.bitwise_or(image, fully_annotated_image)
    # GAUSSIAN BLUR AND CANNY EDGE
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    edges = cv2.Canny(blurred, 1, 50)  

    # ----------------------FIND THE ANNOTATED AREA TO BE REMOVED----------------------------#
    # CALCULATE EDGES OF ANNOTATIONS
    masked_image = np.zeros_like(original_image) 
    row_with_max_white_pixels = np.argmax(np.sum(edges == 255, axis=1))
    column_with_max_white_pixels = np.argmax(np.sum(edges == 255, axis=0))
    # HORIZONTAL LINE EXTREME POINTS
    start_x = 0
    end_x = image.shape[1] - 1
    y = row_with_max_white_pixels
    # VERTICAL LINE EXTREME POINTS
    start_y = 0
    end_y = image.shape[0] - 1
    x = column_with_max_white_pixels
    horizontal_line_length = end_x - start_x
    vertical_line_length = end_y - start_y
    # DETERMINE BOX DIRECTION ACCORDING TO INTERSECTION POINT
    if horizontal_line_length / 2 < x - start_x:
        end_x = x
    else:
        start_x = x

    if vertical_line_length / 2 < y - start_y:
        end_y = y
    else:
        start_y = y
    # DRWAW HORIZONTAL LINE
    cv2.line(masked_image, (start_x, y), (end_x, y), (255, 255, 255), 1)
    # DRAW VERTICAL LINE
    cv2.line(masked_image, (x, start_y), (x, end_y), (255, 255, 255), 1)
    # FILL THE RECTANGLE WITH WHITE PIXELS
    count = vertical_line_length
    while count:
        y += 1
        cv2.line(masked_image, (start_x, y), (end_x, y), (255, 255, 255), 1)
        count -= 1

    # --------------------------REMOVE ANNOTATED AREA FROM FULLY ANNOTATED IMAGE ------------#
    result1 = cv2.bitwise_or(masked_image, original_image)

    # --------------------------EXTRACT THE UNANNOTATED PART FROM THE ORIGINAL IMAGE---------------#
    inverted_image = cv2.bitwise_not(masked_image)
    result2 = cv2.bitwise_or(inverted_image, fully_annotated_image)

    # -------------------------ADD THE UNANNOTATED PART OF ORIGINAL IMAGE TO FULLY ANNOTATED IMAGE TO GET PARTIALLY ANOTATED IMAGE------#
    partially_annotated_image = cv2.bitwise_and(result1, result2)
    cv2.imshow("partially annotated image", partially_annotated_image)
    cv2.imwrite(partially_annotated_image_path, partially_annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


original_path = r"C:\Users\91939\OneDrive\Desktop\My Placement\Companies Applied\Guvi_Companies\Akaike_Data_Scientist\2_Task_CV\internship-assignment-cv-main\Dataset\data\original_image4.jpg"
fully_annotated_path = r"C:\Users\91939\OneDrive\Desktop\My Placement\Companies Applied\Guvi_Companies\Akaike_Data_Scientist\2_Task_CV\internship-assignment-cv-main\Dataset\data\fully_annotated_image4.jpg"
partially_annotated_path = r"partially_annotated.jpg"

get_output_image(original_path, fully_annotated_path, partially_annotated_path)
