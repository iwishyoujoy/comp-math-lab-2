from prettytable import PrettyTable
from sympy import *
from sympy.abc import x, y
import matplotlib.pyplot as plt
import numpy as np
from art import *
import sympy
import math

def validate_interval(a, b):
    if (a >= b):
        print("Введены неверные границы интервала")
        return False
    return True

def plot_function(equation, a, b):
    # преобразуем строку уравнения в функцию
    f = lambda x: eval(equation)

    # создаем массив значений x на заданном промежутке c запасом в единичку
    x = np.linspace(a-1, b+1, 1000)

    # создаем массив значений y
    y = f(x)

    # строим график функции
    plt.plot(x, y)

    # добавляем оси координат
    plt.axhline(y=0, color='k', lw=0.5)
    plt.axvline(x=0, color='k', lw=0.5)

    # добавляем заголовок и подписи осей
    plt.title(f"График функции f(x) = {equation}")
    plt.xlabel("x")

# def plot_functions(equation1, equation2):
#     # создаем символьные переменные x и y
#     x, y = sympy.symbols('x y')

#     # преобразуем строки уравнений в выражения sympy
#     f1 = sympy.sympify(equation1)
#     f2 = sympy.sympify(equation2)

#     # создаем функции для вычисления значений f1(x) и f2(x)
#     f1_func = sympy.lambdify(x, f1.subs(y, x), modules=['numpy', 'math'])
#     f2_func = sympy.lambdify(x, f2.subs(y, x), modules=['numpy', 'math'])

#     # создаем массив значений x на заданном промежутке
#     x_vals = np.linspace(-10, 10, 1000)

#     # создаем массив значений y для первой функции
#     y1_vals = f1_func(x_vals)

#     # строим график первой функции
#     plt.plot(x_vals, y1_vals, label=f"f1(x) = {equation1}")

#     # создаем массив значений y для второй функции
#     y2_vals = f2_func(x_vals)

#     # строим график второй функции
#     plt.plot(x_vals, y2_vals, label=f"f2(x) = {equation2}")

#     # добавляем оси координат
#     plt.axhline(y=0, color='k', lw=0.5)
#     plt.axvline(x=0, color='k', lw=0.5)

#     # добавляем заголовок и подписи осей
#     plt.title(f"Графики функций f1(x) = {equation1} и f2(x) = {equation2}")
#     plt.xlabel("x")
#     plt.ylabel("f(x)")

#     # добавляем легенду
#     plt.legend()

#     # выводим график на экран
#     plt.show()

def plot_system(equation_1, equation_2, a, b):
    f1 = lambda x: eval(equation_1)
    f2 = lambda x: eval(equation_2)
    x = np.linspace(a, b, 1000)
    y1 = f1(x)
    y2 = f2(x)
    
    plt.plot(x, y1, label=f"y1 = {equation_1}")
    plt.plot(x, y2, label=f"y2 = {equation_2}")
    
    plt.axhline(y=0, color='k', lw=0.5)
    plt.axvline(x=0, color='k', lw=0.5)
    
    plt.title("График системы уравнений")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    
    plt.show()


def bisection_method(eq, a, b, epsilon, max_iter):
    tprint("Bisection  method")
    # преобразуем уравнение в функцию
    f = lambdify(x, eq)

    # вычисляем значение функции в точках a и b
    fa = f(a)
    fb = f(b)

    # проверяем, что на заданном промежутке есть только один корень
    if fa*fb > 0:
        print("На заданном промежутке нет единственного корня")
    else:
        # применяем метод половинного деления
        i = 1
        table = PrettyTable()
        table.field_names = ["№ итерации", "a", "b", "x", "f(a)", "f(b)", "f(x)", "|a-b|"]
        while (b - a > epsilon or fc >= epsilon) and (i <= max_iter):
            c = (a + b)/2
            fc = f(c)
            table.add_row([i, a, b, c, fa, fb, fc, abs(a-b)]) #вывод строки можно поставить и после i+=1
            if fa*fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
            i += 1
        # выводим найденный корень
        table.add_row([i, a, b, c, fa, fb, fc, abs(a-b)])
        print(table)
        if i > max_iter:  # Добавлено условие для вывода сообщения о превышении лимита итераций
            print("Количество итераций превысило лимит")
        else:
            # Выводим найденный корень
            print("Найденный корень: ", c)

def newton_method(eq, a, b, epsilon, max_iter):
    tprint("Newton  method")
    # преобразуем уравнение в функцию
    f = lambdify(x, eq)

    # выбираем начальное приближение
    if f(a) == 0:
        x0 = a
    elif f(b) == 0:
        x0 = b
    elif f(a) * f(b) > 0:
        print("Невозможно выбрать начальное приближение")
        return
    elif f(a) * f(b) < 0:
        x0 = a

    # вычисляем значение функции и ее производной в начальной точке
    fx = f(x0)
    fx_deriv = (f(x0 + epsilon) - f(x0)) / epsilon

    # проверяем, что производная не равна нулю в начальной точке
    if fx_deriv == 0:
        print("Производная равна нулю в начальной точке")
    else:
        # применяем метод Ньютона
        i = 1
        table = PrettyTable()
        table.field_names = ["№ итерации", "x", "f(x)", "f'(x)", "|x - x_prev|"]
        while (abs(fx) > epsilon) and (i <= max_iter):
            x_prev = x0
            x0 = x0 - fx/fx_deriv
            fx = f(x0)
            fx_deriv = (f(x0 + epsilon) - f(x0)) / epsilon
            table.add_row([i, x0, fx, fx_deriv, abs(x0-x_prev)])
            i += 1
        # выводим найденный корень
        table.add_row([i, x0, fx, fx_deriv, abs(x0-x_prev)])
        print(table)
        if i > max_iter:  # Добавлено условие для вывода сообщения о превышении лимита итераций
            print("Количество итераций превысило лимит")
        else:
            # Выводим найденный корень
            print("Найденный корень: ", x0)

def simple_iteration(eq, a, b, epsilon, max_iter):
    tprint("Simple  iteration  method")

    f = sympify(eq)
    x = symbols('x')
    # задаем функцию g(x) для метода простой итерации, например, g(x) = x - 0.5 * f(x) / f'(x)
    g = x - 0.5 * f/ diff(f, x)

    # выбираем начальное приближение ближе к корню
    c0 = (a+b)/2

    # инициализируем переменные
    c_prev = c0
    c = lambdify(x, g, modules=['math'])(c0)
    i = 1

    # проверяем условие сходимости
    g_deriv = diff(g, x)
    g_deriv_fn = lambdify(x, g_deriv, modules=['math'])
    if max(abs(g_deriv_fn(a)), abs(g_deriv_fn(b))) >= 1:
        print("Метод не сходится на заданном интервале")
        return

    # создаем таблицу для вывода результатов
    table = PrettyTable()
    table.field_names = ["№ итерации", "x", "f(x)", "|x - x_prev|"]

    # выполняем итерации
    while abs(c - c_prev) > epsilon and i <= max_iter:
        c_prev = c
        c = lambdify(x, g, modules=['math'])(c_prev)
        f_val = lambdify(x, f, modules=['math'])(c)
        table.add_row([i, c, f_val, abs(c-c_prev)])
        i += 1

    # выводим результаты
    print(table)
    if i > max_iter:
        print("Количество итераций превысило лимит")
    else:
        print("Найденный корень: ", c)

def newton_method_for_system(eq1, eq2, x0, y0, epsilon, max_iter):
    
    tprint("Newton  method")
    # преобразуем уравнения в функции
    x, y = symbols('x y')
    f1 = lambdify((x, y), eq1, modules=['math'])
    f2 = lambdify((x, y), eq2, modules=['math'])

    # задаем начальное приближение
    x_curr, y_curr = x0, x0
    x_prev, y_prev = x0, y0

    # определяем матрицу Якоби и ее обратную
    J = Matrix([[diff(eq1, x), diff(eq1, y)], [diff(eq2, x), diff(eq2, y)]])
    J_inv = J.inv()

    # применяем метод Ньютона
    i = 1
    table = PrettyTable()
    table.field_names = ["№ итерации", "x", "y", "f1(x,y)", "f2(x,y)", "|x - x_prev|", "|y - y_prev|"]
    while True:
        # вычисляем значения функций и их производных в текущей точке
        fx = f1(x_prev, y_prev)
        fy = f2(x_prev, y_prev)
        fx_deriv, fy_deriv = J_inv.subs([(x, x_prev), (y, y_prev)]).tolist()

        # вычисляем приращения
        dx, dy = fx_deriv[0]*fx + fx_deriv[1]*fy, fy_deriv[0]*fx + fy_deriv[1]*fy

        # обновляем значения x и y
        x_curr = x_prev - dx
        y_curr = y_prev - dy

        # проверяем критерий окончания итерационного процесса
        if (abs(x_curr - x_prev) <= epsilon and abs(y_curr - y_prev) <= epsilon) or i >= max_iter:
            break

        # добавляем результаты текущей итерации в таблицу
        table.add_row([i, x_curr, y_curr, fx, fy, abs(x_curr - x_prev), abs(y_curr - y_prev)])

        # обновляем значения x_prev и y_prev
        x_prev, y_prev = x_curr, y_curr

        # увеличиваем счетчик итераций
        i += 1

    # выводим результаты
    table.add_row([i, x_curr, y_curr, fx, fy, abs(x_curr - x_prev), abs(y_curr - y_prev)])
    print(table)
    print(f"Найденные корни: x={x_curr}, y={y_curr}")

def equation_solve():
    # РЕШЕНИЕ НЕЛИНЕЙНЫХ УРАВНЕНИЙ
    max_iter = 50;

    # ввод данных
    equation = input("Введите уравнение: ")
    a = float(input("Введите левую границу интервала: "))
    b = float(input("Введите правую границу интервала: "))
    if (not validate_interval(a, b)): exit 
    epsilon = float(input("Введите погрешность вычисления: "))

    # использование методов
    bisection_method(equation, a, b, epsilon, max_iter)
    newton_method(equation, a, b, epsilon, max_iter)
    simple_iteration(equation, a, b, epsilon, max_iter)

    # вывод графика для функции (не работает с трансцендентными функциями)
    # plot_function(equation, a, b)

def system_of_equations_solve():
    # РЕШЕНИЕ СИСТЕМ НЕЛИНЕЙНЫХ УРАВНЕНИЙ
    max_iter = 50;
    
    # ввод данных
    equation1 = input("Введите 1-ое уравнение: ")
    equation2 = input("Введите 2-ое уравнение: ")

    # вывод графика для функций (не работает, к сожалению)
    # plot_system(equation1, equation2, -10, 10)

    # ввод остальных данных
    x0 = float(input("Введите приближение для X: "))
    y0 = float(input("Введите приближение для Y: "))
    epsilon = float(input("Введите погрешность вычисления: "))

    # использование методов
    newton_method_for_system(equation1, equation2, x0, y0, epsilon, max_iter)

# необходиимо выбрать и раскомментировать что-то одно
# equation_solve()
system_of_equations_solve()


    



