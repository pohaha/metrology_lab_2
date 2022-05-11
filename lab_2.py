
import math
from pylab import *
from scipy.linalg import *
from matplotlib import pyplot as plt
from rows import polynomial_row
from rows import superscript_numbers


def float_range(start, stop, step):
  while start <= stop:
    yield float(start)
    start += step
    start = round(start,2)

def create_starting_data(input_x_values):
    y_values = []
    for element in input_x_values:
        y_values.append(round(base_function(element), 4))
    return y_values
    

def print_table_data(input_x_values, input_y_values):
    if len(input_x_values) != len(input_y_values):
        print("ERROR UNEVEN X AND Y ARRAYS")
        return
    for index in range(len(input_x_values)):
        print(F"[{index}]\t X = {input_x_values[index]}, Y = {input_y_values[index]}")

#INITIATING STARTING DATA
def var_six_func(x_value):
    return x_value * math.sin(x_value)
x_values = list(float_range(0, 6, 0.1))
base_function = var_six_func
y_values = create_starting_data(x_values)
print_table_data(x_values, y_values)

#LEAST SQUARE FIT MODEL
def super_each(input_values, power):
    rt_array = []
    for element in input_values:
        rt_array.append(element**power)
    return rt_array

# задаем вектор m = [x**3, x**2, x, E]
m=vstack((super_each(x_values, 3), super_each(x_values, 2),x_values,ones(len(x_values)))).T

# находим коэффициенты при составляющих вектора m
s=lstsq(m,y_values)[0]

# на отрезке [-5,5]
x_prec=linspace(0, 6,101)

# рисуем точки
plot(x_values,y_values,'D')

# рисуем кривую вида y = ax^3+bx^2 + cx + d, подставляя из решения коэффициенты s[0], s[1], s[2], s[3]
plot(x_prec,s[0]*x_prec**3+s[1]*x_prec**2+s[2]*x_prec+s[3],'-',lw=1)
legend(('Data','Least square fit'))
grid()
plt.show()

# LINEAR INTERPOLATION MODEL

x_lin_interpolation = np.linspace(0, 6, 120)
y_lin_interpolation = np.interp(x_lin_interpolation, x_values, y_values)
plot(x_values,y_values,'D')
plot(x_lin_interpolation, y_lin_interpolation, '.-', lw=1)
legend(('Data','linear interpolation'))
grid()
plt.show()


# LAGRANGE POLYNOMIAL INTERPOLATION MODEL
def count_base_polynomial(current_index: int, x_values: list, new_x: float):
    base_polynomial_value = 1
    for index in range(size(x_values)):
        if index != current_index:
            base_polynomial_value *= (new_x - x_values[index]) / (x_values[current_index] - x_values[index])
    return base_polynomial_value

def lagrange_interpolation(new_x: float, known_x: list, known_y: list):
    new_y = 0
    if(size(known_x) != size(known_y)):
        print("ERROR occured during lagrange interpolation: inputted x and y arrays are of different size")
        return
    for index in range(size(known_x)):
        new_y += known_y[index] * count_base_polynomial(index, known_x, new_x)
    return new_y

plot(x_values,y_values,'.-')
current_x_value = 0
data_x = []
data_y = []

while current_x_value < 5:
    current_x_value += 0.1
    data_x.append(current_x_value)
    data_y.append(lagrange_interpolation(current_x_value, x_values, y_values))
plot(data_x, data_y, "D")
print(data_y)
legend(('Data','lagrange interpolation'))
plt.show()
grid()

# CUBIC SPLINE INTERPOLATION MODEL

#split main interval in 2 intervals (main = [x_min, x_max], new = { [ x_min, x_min+(x_max-x_min)/2 ], 
#                                                                   [ x_min+(x_max-x_min)/2, x_max ] })
#                                                           delta = (x_max-x_min) / 6
#add 4 conditions in each interval { (x1, y(x1)),           x1 = x_min
#                                    (x2, y(x2)),           x2 = x1 + delta   
#                                    (x3, y(x3)),           x3 = x2 + delta   
#                                    (x4, y(x3)),           x4 = x3 + delta   
#                                    (x5, y(x3)),           x5 = x4 + delta   
#                                    (x6, y(x3)),           x6 = x5 + delta   
#                                    (x7, y(x3)) }          x7 = x6 + delta

# count polynomial^3 #1:
#       S0(x) = x^3*a3 + x^2*a2 + x*a1 + a0 => 
#      -------------------------------------------------------
#      |    y(x1) = (x1^3 * a3) + (x1^2 * a2) + (x1 * a1) + a0
#      /    y(x2) = (x2^3 * a3) + (x2^2 * a2) + (x2 * a1) + a0
#      \    y(x3) = (x3^3 * a3) + (x3^2 * a2) + (x3 * a1) + a0
#      |    y(x4) = (x4^3 * a3) + (x4^2 * a2) + (x4 * a1) + a0
#      -------------------------------------------------------

# count polynomial^3 #2:
#       S1(x) = x^3*a3 + x^2*a2 + x*a1 + a0 => 
#      -------------------------------------------------------
#      |    y(x4) = (x4^3 * a3) + (x4^2 * a2) + (x4 * a1) + a0
#      /    y(x5) = (x5^3 * a3) + (x5^2 * a2) + (x5 * a1) + a0
#      \    y(x6) = (x6^3 * a3) + (x6^2 * a2) + (x6 * a1) + a0
#      |    y(x7) = (x7^3 * a3) + (x7^2 * a2) + (x7 * a1) + a0
#      -------------------------------------------------------

class spline_polynomial:
    range_start = 0
    range_end = 0
    conditions = {}
    rows = []
    name = ""
    final_koefficients = []


    def count_koefficient(self, current_koefficient_id = "start"):
        # A0 = value0 / X[0,0]
        # A1 = (value1 - A0*X[1,0])/ X[1,1]
        # A2 = (value2 - A0*X[2,0] - A1*X[2,1]) / X[2,2]
        # A3 = (value3 - A0*X[3,0] - A1*X[3,1] - A2*X[3,2]) / X[3,3]
        if (current_koefficient_id == "start"):
            current_koefficient_id = len(self.final_koefficients) - 1
        koefficient_value = self.rows[current_koefficient_id].polynomial_value
        for component_id in range(current_koefficient_id):
            if(self.final_koefficients[component_id] == "undefined"):
                self.count_koefficient(component_id)
            koefficient_value -= self.final_koefficients[component_id] * self.rows[current_koefficient_id][component_id]
        koefficient_value /= self.rows[current_koefficient_id][current_koefficient_id]
        self.final_koefficients[current_koefficient_id] = koefficient_value

    def __init__(self, x_values: float, y_values: float, name = "Undefined"):
        self.rows = []
        self.conditions = {}
        self.final_koefficients = []
        self.range_start = x_values[0]
        self.range_end = x_values[-1]
        self.name = name
        row_id = 0
        ## fill in starting rows
        for element in x_values:
            koefficients = []
            self.conditions[element] = y_values[row_id]
            self.final_koefficients.append("undefined")
            col_id = 0
            for power_value in range(len(x_values)):
                koefficients.append(pow(element, power_value))
                col_id += 1
            self.rows.append(polynomial_row(koefficients, y_values[row_id]))
            row_id += 1

        ## getting SLAE to a triangular form
        for row in self.rows:
            print(F"\t\t{str(row)}")
        for col_id in range(len(self.rows)-1, 0,-1):
            for row_id in range(col_id):
                self.rows[row_id] = self.rows[row_id] * self.rows[col_id][col_id] - (self.rows[col_id] * self.rows[row_id][col_id])
                print(F"\t{self.rows[col_id][col_id]} * R{row_id} - {self.rows[row_id][col_id]} * R{col_id},")
            for row in self.rows:
                print(F"\t\t{str(row)}")
        #counting final coefficients of spline polynomial
        self.count_koefficient()
        
        


        
    def show(self):
        print(F"Polynomial: {self.name}")
        print(F"active in range [{self.range_start}, {self.range_end}]")
        print(F"{self.name}(X) =", end = " ")
        index = 0
        for koefficient in self.final_koefficients:
            final_symbol = "+"
            if index == len(self.final_koefficients) - 1:
                final_symbol = ""
            print(F"{koefficient}*X{superscript_numbers[index]} {final_symbol}", end = " ")
            index += 1

        print("\ncounted from:")
        for row in self.rows:
            print(F"\t{str(row)}")
        print("as derived from conditions: ")
        for key in self.conditions.keys():
            print(F"\t{self.name}({key}) = {self.conditions[key]}")
        

    def count_for(self, x_value: float):
        result = 0
        for i in range(len(self.final_koefficients)):
            result += pow(x_value, i) * self.final_koefficients[i]
        return result

    def holds(self, x_value: float):
        if(x_value >= self.range_start) and (x_value <= self.range_end):
            return True
        return False


class spline:
    first_half = 0
    second_half = 0
    def __init__(self, x_values: list, y_values: list):
        centre = round(len(x_values) / 2)
        self.first_half = spline_polynomial(x_values[0:centre], y_values[0:centre], "S0")
        self.second_half = spline_polynomial(x_values[centre-1:], y_values[centre-1:], "S1")
    def show(self):
        print("first half of numeric range:")
        self.first_half.show()
        print("second half of numeric range:")
        self.second_half.show()
    
    def count_for(self, new_x_value: float):
        if(self.first_half.holds(new_x_value)):
            return self.first_half.count_for(new_x_value)
        if(self.second_half.holds(new_x_value)):
            return self.second_half.count_for(new_x_value)
        print("ERROR! value is not in range")
        return False

# create new x array with conditions for spline interpolation (4 for one half [0..3] and 4 for other half[3..6])
spline_interpolation_x_array = [0,1,2,3,4,5,6]
spline_interpolation_y_array = create_starting_data(spline_interpolation_x_array)
test_spline = spline(spline_interpolation_x_array, spline_interpolation_y_array)
test_spline.show()
spline_interpolation_y_values = []
for x_value in x_values:
    spline_interpolation_y_values.append(test_spline.count_for(x_value))

plot(x_values,y_values,'.-')
plot(x_values, spline_interpolation_y_values, "D")
legend(("Data", "spline_interpolation"))
plt.show()
