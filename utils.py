import numpy as np
import cv2
import tensorflow as tf


class HandwrittenCharacterDetector:
    """
    HandwrittenCharacterDetector receives an image in a form of a numpy array and for each detected
    character it should return its coordinates, and a cropped image.
    """

    def __init__(self, img):
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.img = img
        if np.mean(self.img) > 127:
            # Assuming most of the image is background, if the background is light the image 
            # is inverted so that the background is dark and the handwritten text is bright
            self.img = 255 - self.img
        self.img_h, self.img_w = self.img.shape[:2]
        (self.thresh, self.im_bw) = cv2.threshold(self.img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        self.contours, _ = cv2.findContours(self.im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    def getCroppedImages(self):
        """
        Extract character images from input image
        Return:
            result (list of dict[img: numpy.array, coords: tuple]) : List of dicts where each dict 
                                                                     has image of character and its coords
        """

        result = []
        for contour in self.contours:
            w_min, h_min = np.min(contour, axis=0)[0]
            w_max, h_max = np.max(contour, axis=0)[0]
            height = h_max - h_min
            width = w_max - w_min
            if width * height > 100:
                cropped_img = np.zeros((height+20, width+20), dtype=np.uint8)
                con = contour.reshape(-1, 2)
                con[:, 0] -= np.min(con[:, 0]) - 10
                con[:, 1] -= np.min(con[:, 1]) - 10
                cv2.fillPoly(cropped_img, pts =[con], color=(255))
                result.append({
                    "img": cropped_img,
                    "coords": ((h_min, w_min), (h_max, w_max))
                })

        return sorted(result, key = lambda i: i['coords'][0][1])

class HandwrittenCharacterClassifier:
    """
    HandwrittenCharacterClassifier receives an image of the character
    and return the corresponding label.
    """

    id2label = {
        0: '0',
        1: '1',
        2: '2',
        3: '3',
        4: '4',
        5: '5',
        6: '6',
        7: '7',
        8: '8',
        9: '9',
        10: '+',
        11: '-',
        12: 'x',
        13: '/',
        14: '(',
        15: ')',
    }
    model = tf.keras.models.load_model('./model/model.h5')
    
    def __init__(self, img):
        self.img = img
        
    @staticmethod
    def rotate_image(img, angle):
        """
        Args:
            img (numpy.array): Image
            angle (int): Angle at which the input image should be rotated
        Returns:
            result (numpy.array): Rotated image
        """

        image_center = tuple(np.array(img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    @staticmethod
    def padImg2Square(img):
        """
        Creates a square image from the input image
        Args:
            img (numpy.array): Image
        Returns:
            result (numpy.array): Squared image
        """

        height, width = img.shape
        top = bottom = max(0, (width - height) // 2)
        left = right = max(0, (height - width) // 2)
        padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)
        return padded_img
    
    def getLabel(self):
        """
        Converts input image to corresponding label.
        Returns:
            label (str): Corresponding label 
            confidence (float): Confidence of prediction (from 0.0 to 1.0)
        """

        preprocessed_img = cv2.resize(HandwrittenCharacterClassifier.padImg2Square(self.img), (28, 28))
        prediction = tf.nn.softmax(HandwrittenCharacterClassifier.model.predict(preprocessed_img[np.newaxis, :, :, np.newaxis]))[0]
        probability = np.max(prediction)
        
        if probability < 0.95:
            new_predictions_array, new_predictions = [], []
            for angle in [-10, -5, 5, 10]:
                aug_image = preprocessed_img.copy()
                aug_image = HandwrittenCharacterClassifier.rotate_image(aug_image, angle)
                new_prediction_array = tf.nn.softmax(HandwrittenCharacterClassifier.model.predict(aug_image[np.newaxis, :, :, np.newaxis]))[0]
                new_predictions_array.append(new_prediction_array)
                new_prediction = np.max(new_prediction_array)
                if new_prediction > 0.95:
                    prediction = new_prediction_array
                    break
                new_predictions.append(new_prediction)
            else:
                prediction = new_predictions_array[np.argmax(new_predictions)]
        
        label = HandwrittenCharacterClassifier.id2label[np.argmax(prediction)]
        confidence = np.max(prediction)
        return label, confidence

class Solver:
    """
    Solver receives an string of the mathematical expression
    and return the solution.
    """

    def __init__(self, expression):
        self.expression = expression
        self.result = Solver.getFinalResults(expression)
    
    @staticmethod
    def findSubEx(ex):
        """
        Finds indexes of sub
        Args:
            ex (str): Image
        Returns:
            result (numpy.array): Squared image
        """

        start_index, end_index = 0, len(ex)
        end_index = len(ex)
        for index, char in enumerate(ex):
            if char == "(":
                old_start_index = start_index
                start_index = index + 1
            elif char == ")":
                end_index = index
                try:
                    int(ex[start_index:end_index])
                    start_index, end_index = old_start_index, len(ex)
                    continue
                except:
                    pass
                return start_index, end_index
        if start_index != 0:
            raise ValueError("The brackets must be closed")
        return None
    
    @staticmethod
    def operation(a, b, operator):
        """
        Performs a mathematical operation with input numbers a and b
        Args:
            a (int): First operand
            b (int): Second operand
            operator (str): Operation (+, -, x or /)
        Returns:
            result (int): Result of the operation
        """

        if operator == "+":
            return a+b
        elif operator == "-":
            return a-b
        elif operator == "*":
            return a*b
        elif operator == "/":
            return a/b
        else:
            raise ValueError("Parameter operator must have one of the following values */+-")

    @staticmethod
    def getFinalScore(numbers, operators):
        """
        Performs mathematical operations with input numbers
        Args:
            numbers (list of int): List of input numbers (size N)
            operators (list of str): List of input operations (size N-1)
        Returns:
            result (int): Result of the operations
        """

        if len(numbers) == 1:
            return numbers[0]
        else:
            prod_index = operators.index("*") if "*" in operators else float("inf")
            divide_index = operators.index("/") if "/" in operators else float("inf")

            index = min(prod_index, divide_index)
            if index == float("inf"):
                plus_index = operators.index("+") if "+" in operators else float("inf")
                minus_index = operators.index("-") if "-" in operators else float("inf")
                index = min(plus_index, minus_index)

            number_1, number_2 = numbers[index], numbers[index+1]
            operator = operators[index]
            new_number = Solver.operation(number_1, number_2, operator)
            numbers[index] = new_number
            del operators[index]
            del numbers[index+1]
            return Solver.getFinalScore(numbers, operators)

    @staticmethod
    def calculateWithoutBrackets(ex):
        """
        Calculates the value of the expression provided there are no parentheses 
        (it is possible that there is only a negative number in parentheses e.g. (-10)).
        Args:
            ex (str): Expression
        Returns:
            result (int): Result of the expression
        """

        numbers, operators = [], []
        start_index = 0
        ind = True
        for index, char in enumerate(ex):
            if char == "(":
                ind = False
            elif char == ")":
                ind = True

            if (char in "*/+-" and ind) or index == len(ex)-1:
                end_index = index if index != len(ex) -1 else None
                number = ex[start_index:end_index].replace("(", "").replace(")", "")
                number = int(number)
                numbers.append(number)
                if index != len(ex) -1:
                    operators.append(char)
                start_index = index + 1
        return Solver.getFinalScore(numbers, operators)

    @staticmethod
    def getFinalResults(ex):
        """
        Calculates the value of the expression.
        Args:
            ex (str): Expression
        Returns:
            result (int): Result of the expression
        """

        indexes = Solver.findSubEx(ex)
        if indexes is None:
            return Solver.calculateWithoutBrackets(ex)
        else:
            ex_without_brackets = ex[indexes[0]:indexes[1]]
            new_value = Solver.calculateWithoutBrackets(ex_without_brackets)
            new_value = str(new_value) if new_value > 0 else f"({new_value})"
            new_ex = ex[:indexes[0]-1] + new_value + ex[indexes[1]+1:]
            return Solver.getFinalResults(new_ex)

class ExpressionGenerator:
    """
    ExpressionGenerator Generates mathematical expressions
    """

    sign_list = ["+", "-", "*", "/"]

    def __init__(self, number_of_expressions):
        self.number_of_expressions = number_of_expressions

    @staticmethod
    def generateExpression():
        """
        Generates mathematical expressions without parentheses
        Returns:
            result (str): Mathematical expression
        """

        number_of_operatoins = np.random.randint(3, 10)
        numbers = np.random.randint(-1000, 1000, size=number_of_operatoins+1).astype(str)

        sign_pos = np.random.randint(3, size=number_of_operatoins)
        signs = [ExpressionGenerator.sign_list[sign] for sign in sign_pos]

        ex = ""
        for number, sign in zip(numbers, signs):
            if int(number) < 0:
                number = f"({number})"
            ex += number + sign
        if int(numbers[-1]) < 0:
            ex += f"({numbers[-1]})"
        else:
            ex += numbers[-1]
        return ex

    @staticmethod
    def mergeMultipleExpression():
        """
        Merge multiple mathematical expressions without parentheses
        Returns:
            result (str): Mathematical expression
        """
        number_of_ex = np.random.randint(1, 4)
        if number_of_ex == 1:
            return ExpressionGenerator.generateExpression()
        sign_pos = np.random.randint(1, 4, size=number_of_ex-1)
        
        signs = [ExpressionGenerator.sign_list[sign] for sign in sign_pos]
        ex = ""
        for index in range(number_of_ex-1):
            ex += f"({ExpressionGenerator.generateExpression()}){signs[index]}"
        ex += f"({ExpressionGenerator.generateExpression()})"
        return ex

    def generateExpression(self):
        for i in range(self.number_of_expressions):
            yield ExpressionGenerator.mergeMultipleExpression()


def testExpressionGenerator(num_of_tests=1000):
    """
    A simple test for the ExpressionGenerator class
    """
    ex_gen = ExpressionGenerator(num_of_tests)
    for generated_ex in ex_gen.generateExpression():
        assert eval(generated_ex) == Solver.getFinalResults(generated_ex)


def getResult(img_path):
    """
    A method that will return a solution for the input image
    Args:
            img_path (str): Path to the image
        Returns:
            solution (int): Solution
            expression (str): Expression
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    hcd = HandwrittenCharacterDetector(img)
    cropped_images = hcd.getCroppedImages()

    expression = ''
    for cropped_image in cropped_images:
        hcc = HandwrittenCharacterClassifier(cropped_image['img']/255)
        character, confidence = hcc.getLabel()
        expression += character

    solution = Solver.getFinalResults(expression)
    return solution, expression