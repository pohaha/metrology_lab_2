superscript_numbers = ["⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹"]
subscript_numbers = ["₀","₁","₂","₃","₄","₅","₆","₇","₈","₉"]
class polynomial_row:
    koefficients = []
    polynomial_value = 0
    def __init__(self, koefficients: list, function_value: float):
        self.koefficients = []
        for value in koefficients:
            self.koefficients.append(value)
        self.polynomial_value = function_value
    def __add__(self, other):
        if type(other) != polynomial_row:
            print("ERROR, add operation not on polynomial row class")
            return False
        if len(self.koefficients) != len(other.koefficients):
            print("ERROR sizes do not match")
            return False
        new_koefficients = self.koefficients.copy()
        new_polynomial_value = self.polynomial_value
        id = 0
        for element in other.koefficients:
            new_koefficients[id] += element
            id += 1
        new_polynomial_value += other.polynomial_value
        return polynomial_row(new_koefficients, new_polynomial_value)
    
    def __mul__(self, mul_value):
        new_koefficients = self.koefficients.copy()
        new_polynomial_value = self.polynomial_value
        id = 0
        for element in new_koefficients:
            new_koefficients[id] *= mul_value
            id += 1
        new_polynomial_value *= mul_value
        return polynomial_row(new_koefficients, new_polynomial_value)
    
    def __sub__(self, other):
        if type(other) != polynomial_row:
            print("ERROR, add operation not on polynomial row class")
            return False
        if len(self.koefficients) != len(other.koefficients):
            print("ERROR sizes do not match")
            return False
        return self.__add__((other * -1))

    def __str__(self):
        index = 0
        final_string = ""
        for element in self.koefficients:
            end_symbol = "+"
            if index == (len(self.koefficients) - 1):
                end_symbol = "="
            final_string +=str(F"{element}*A{subscript_numbers[index]} {end_symbol} ")
            index += 1
        final_string += str(F"{self.polynomial_value}")
        return final_string
    
    def __getitem__(self, key: int):
        if(key >= len(self.koefficients)):
            print(F"ERROR: no such key ({key}) in polynomial row")
            return -1
        return self.koefficients[key]

    def show(self):
        index = 0
        for element in self.koefficients:
            end_symbol = "+"
            if index == (len(self.koefficients) - 1):
                end_symbol = "="
            print(F"{element}*A{subscript_numbers[index]} {end_symbol}", end = " ")
            index += 1
        print(F"{self.polynomial_value}")
