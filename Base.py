from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import math
from functools import reduce
from operator import mul
import sympy as sp
from sympy import symbols, sympify, series, init_printing
import re
import plotly.graph_objs as go
import plotly.io as pio



def factores_primos_con_potencias_incrementadas(num):
    factores = {}
    divisor = 2

    while divisor * divisor <= num:
        while num % divisor == 0:
            if divisor in factores:
                factores[divisor] += 1
            else:
                factores[divisor] = 1
            num //= divisor
        divisor += 1

    if num > 1:  # Si el número restante es mayor que 1, es un factor primo
        factores[num] = 1

    # Incrementar cada potencia en 1 y calcular el producto de las potencias incrementadas
    potencias_incrementadas = {k: v + 1 for k, v in factores.items()}
    producto_de_potencias_incrementadas = reduce(mul, potencias_incrementadas.values(), 1)

    return potencias_incrementadas, producto_de_potencias_incrementadas

# Función para verificar si un número es primo
def es_primo(num):
    if num <= 1:
        return False
    if num == 2:
        return True
    if num % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(num)) + 1, 2):
        if num % i == 0:
            return False
    return True
def es_divisor(num):
    divisores = []
    if num <= 1:
        return False
    for j in range(2, num + 1):
        if num % j == 0:
            divisores.append(j)
    
    if len(divisores) < 2:
        raise ValueError(f"No se puede encontrar el penúltimo divisor de {num} porque tiene menos de dos divisores.")
    
    iterador = iter(divisores)
    ultimo = next(iterador)
    penultimo = next(iterador)
    for elemento in iterador:
        penultimo, ultimo = ultimo, elemento
    
    sigma = sum(divisores)
    return divisores, penultimo, sigma

# Función para determinar si un número es par o impar
def tipo_numero(num):
    return "par" if num % 2 == 0 else "impar"

# Función para contar números primos anteriores a un número dado
def contar_primos(num):
    lista_primos = []
    for j in range(2, num):
        if es_primo(j):
            lista_primos.append(j)
    return len(lista_primos), lista_primos

# Función para verificar si un número puede formar un número perfecto de Mersenne
def numeros_mersenne(num):
    if not es_primo(num):
        return False
    mersenne = 2**num - 1
    if es_primo(mersenne):
        perfecto = 2**(num - 1) * mersenne
        return perfecto
    return False
    
    
app = Flask(__name__)
app.secret_key = 'tu_clave_secreta'

ruta_excel = "C:/Users/Cristhian/Desktop/Python/base_de_datos.xlsx"

def guardar_datos_en_excel(datos, ruta_excel):
    try:
        datos.to_excel(ruta_excel, index=False)
        print("Los datos se han guardado en:", ruta_excel)
    except Exception as e:
        print("Error al guardar los datos en el archivo Excel:", e)

def cargar_datos_desde_excel(ruta_excel):
    try:
        if os.path.exists(ruta_excel):
            return pd.read_excel(ruta_excel)
        else:
            print("El archivo no existe. Creando una nueva base de datos vacía...")
            return pd.DataFrame(columns=["ID", "Primer Nombre", "Segundo Nombre", "Primer Apellido", "Segundo Apellido", "Edad", "Ciudad", "Telefono"])
    except Exception as e:
        print("Error al cargar los datos desde el archivo Excel:", e)
        return pd.DataFrame(columns=["ID", "Primer Nombre", "Segundo Nombre", "Primer Apellido", "Segundo Apellido", "Edad", "Ciudad", "Telefono"])

def generar_id_unico():
    while True:
        id_nuevo = random.randint(100000, 999999)
        datos_actuales = cargar_datos_desde_excel(ruta_excel)
        if id_nuevo not in datos_actuales['ID'].values:
            return id_nuevo

def borrar_dato(id_borrar):
    try:
        datos = cargar_datos_desde_excel(ruta_excel)
        if id_borrar in datos['ID'].values:
            datos = datos[datos['ID'] != id_borrar]
            guardar_datos_en_excel(datos, ruta_excel)
            print(f"Se ha eliminado el registro con ID {id_borrar}.")
        else:
            print(f"No se encontró ningún registro con el ID {id_borrar}.")
    except Exception as e:
        print("Error al borrar el dato:", e)


@app.route('/')
def main():
    return render_template('main.html')

@app.route('/index')
def index():
    if session.get('logged_in'):
        # Obtener datos u otros contextos necesarios
        some_id = 123  # Aquí deberías obtener el ID apropiadamente
        return render_template('index.html', some_id=some_id)
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'ext_crluengas' and password == 'Colombia2024':
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            return "Credenciales incorrectas, intente de nuevo."
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('main'))


@app.route('/lotka_volterra', methods=['GET', 'POST'])
def lotka_volterra():
    if request.method == 'POST':
        try:
            # Obtener los parámetros del formulario
            a = float(request.form['a'])
            b = float(request.form['b'])
            c = float(request.form['c'])
            d = float(request.form['d'])
            t0 = float(request.form['t0'])
            x0 = float(request.form['x0'])
            y0 = float(request.form['y0'])
            h = float(request.form['h'])
            muestras = int(request.form['muestras'])

            # Definir las ecuaciones
            f = lambda t, x, y: a * x - b * x * y
            g = lambda t, x, y: -c * y + d * x * y

            # Ejecutar el modelo de Lotka-Volterra
            tabla = Lotka_Volterra(f, g, t0, x0, y0, h, muestras)
            ti = tabla[:, 0]
            xi = tabla[:, 1]
            yi = tabla[:, 2]

            # Crear los gráficos
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            ax1.plot(ti, xi, label='Presa (xi)')
            ax1.plot(ti, yi, label='Depredador (yi)')
            ax1.set_title('Modelo predador-presa')
            ax1.set_xlabel('Tiempo')
            ax1.set_ylabel('Población')
            ax1.legend()
            ax1.grid()

            ax2.plot(xi, yi)
            ax2.set_title('Fase del modelo presa-predador')
            ax2.set_xlabel('Presa (x)')
            ax2.set_ylabel('Depredador (y)')
            ax2.grid()

            # Convertir la figura a una imagen PNG en base64
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url_lotka = base64.b64encode(img.getvalue()).decode('utf8')

            plt.close()  # Cerrar la figura para liberar memoria

            return render_template('lotka_volterra.html', plot_url=plot_url_lotka)
        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('lotka_volterra.html')

def Lotka_Volterra(f, g, t0, x0, y0, h, muestras):
    muestras = int(muestras)
    tamano = muestras + 1
    tabla = np.zeros(shape=(tamano, 3), dtype=float)
    tabla[0] = [t0, x0, y0]
    ti = t0
    xi = x0
    yi = y0
    
    for i in range(1, tamano):
        K1x = h * f(ti, xi, yi)
        K1y = h * g(ti, xi, yi)
        
        K2x = h * f(ti + h, xi + K1x, yi + K1y)
        K2y = h * g(ti + h, xi + K1x, yi + K1y)

        xi = xi + (1/2) * (K1x + K2x)
        yi = yi + (1/2) * (K1y + K2y)
        ti = ti + h
        
        tabla[i] = [ti, xi, yi]

    return tabla



@app.route('/ingresar', methods=['GET', 'POST'])
def ingresar_dato():
    if session.get('logged_in'):
        if request.method == 'POST':
            id = generar_id_unico()
            primer_nombre = request.form['primer_nombre'].capitalize()
            segundo_nombre = request.form['segundo_nombre'].capitalize()
            primer_apellido = request.form['primer_apellido'].capitalize()
            segundo_apellido = request.form['segundo_apellido'].capitalize()
            ciudad = request.form['ciudad'].capitalize()
            edad = int(request.form['edad'])
            telefono = request.form['telefono']
            
            registro = {
                "ID": id,
                "Primer Nombre": primer_nombre,
                "Segundo Nombre": segundo_nombre,
                "Primer Apellido": primer_apellido,
                "Segundo Apellido": segundo_apellido,
                "Edad": edad,
                "Ciudad": ciudad,
                "Telefono": telefono
            }
            
            datos = pd.DataFrame([registro])
            datos_existentes = cargar_datos_desde_excel(ruta_excel)
            datos_actualizados = pd.concat([datos_existentes, datos], ignore_index=True)
            guardar_datos_en_excel(datos_actualizados, ruta_excel)

            return render_template('ingresar_dato.html', mensaje="Datos ingresados correctamente.")
        
        return render_template('ingresar_dato.html')
    else:
        return redirect(url_for('login'))

@app.route('/ver_datos')
def ver_datos():
    if session.get('logged_in'):
        datos_actuales = cargar_datos_desde_excel(ruta_excel)
        return render_template('ver_datos.html', datos=datos_actuales.to_dict(orient='records'))
    else:
        return redirect(url_for('login'))
# Renombrar la función de modificación
def actualizar_dato(id_modificar, nuevo_valor, columna_modificar):
    try:
        datos = cargar_datos_desde_excel(ruta_excel)
        if id_modificar in datos['ID'].values:
            datos.loc[datos['ID'] == id_modificar, columna_modificar] = nuevo_valor
            guardar_datos_en_excel(datos, ruta_excel)
            print(f"El registro con ID {id_modificar} ha sido modificado.")
        else:
            print(f"No se encontró ningún registro con el ID {id_modificar}.")
    except Exception as e:
        print("Error al modificar el dato:", e)

@app.route('/modificar_dato/<int:id>', methods=['GET', 'POST'], endpoint='modificar_dato')
def modificar_dato(id):
    if session.get('logged_in'):
        if request.method == 'POST':
            try:
                columna_modificar = request.form['columna_modificar']
                nuevo_valor = request.form['nuevo_valor']
                actualizar_dato(id, nuevo_valor, columna_modificar)
                return redirect(url_for('ver_datos'))
            except Exception as e:
                print("Error al modificar el dato:", e)
                return "Error al modificar el dato."

        datos = cargar_datos_desde_excel(ruta_excel)
        dato_seleccionado = datos[datos['ID'] == id].to_dict('records')[0]
        columnas_disponibles = datos.columns.tolist()

        return render_template('modificar_dato.html', dato=dato_seleccionado, columnas=columnas_disponibles)
    else:
        return redirect(url_for('login'))


@app.route('/borrar_dato/<int:id>', methods=['POST'])
def borrar_dato_route(id):
    if session.get('logged_in'):
        try:
            borrar_dato(id)
            return redirect(url_for('ver_datos'))
        except Exception as e:
            print("Error al borrar el dato:", e)
            return "Error al borrar el dato."
    else:
        return redirect(url_for('login'))


@app.route('/Facenumber', methods=['GET', 'POST'])
def Facenumber():
    if request.method == 'POST':
        numero = int(request.form['numero'])
        divisores, penultimo, sigma = es_divisor(numero)
        tipo = tipo_numero(numero)
        primo = es_primo(numero)
        perfecto = numeros_mersenne(numero)
        contar, lista_primos = contar_primos(numero)
        potencias, producto = factores_primos_con_potencias_incrementadas(numero) 
        resultado = {
            'potencias':potencias,
            'producto':producto,
            'sigma':sigma,
            'divisor': divisores, 
            'maximo': penultimo, 
            'numero': numero,
            'tipo': tipo,
            'primo': primo,
            'perfecto': perfecto,
            'contar_primos': contar,
            'lista_primos': lista_primos
        }
        
        return render_template('Facenumber.html', resultado=resultado)
    
    return render_template('Facenumber.html')
@app.route('/aspiradora_qr', methods=['GET', 'POST'])
def aspiradora_qr_form():
    error = None
    qr_img_data = None
    matriz = None

    if request.method == 'POST':
        try:
            filas = int(request.form['filas'])
            columnas = int(request.form['columnas'])

            if filas <= 0 or columnas <= 0:
                error = "Los valores de filas y columnas deben ser mayores a cero."
            else:
                # Crear la matriz aleatoria de 0s y 1s
                array = np.random.rand(filas, columnas)
                array = np.round(array).astype(int)

                # Convertir la matriz a una lista de listas para su renderización
                matriz = array.tolist()

                # Crear la figura y los ejes
                fig, ax = plt.subplots()
                ax.matshow(array, cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect('equal')

                # Convertir la figura a una imagen PNG en base64
                img_io = io.BytesIO()
                plt.savefig(img_io, format='png')
                img_io.seek(0)
                qr_img_data = base64.b64encode(img_io.getvalue()).decode('utf8')

                plt.close()  # Cerrar la figura para liberar memoria

        except ValueError:
            error = "Los valores ingresados deben ser números enteros positivos."
        except Exception as e:
            error = f"Se produjo un error al generar la imagen: {e}"

    return render_template('aspiradora_qr_form.html', error=error, qr_img_data=qr_img_data, matriz=matriz)
init_printing(use_latex=False)

# Definimos las variables simbólicas
t, y = symbols('t y')

def preprocess_expression(expr):
    expr = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', expr)  # 2x -> 2*x, 2(x+1) -> 2*(x+1)
    expr = re.sub(r'([a-zA-Z)])(\d)', r'\1*\2', expr)  # x2 -> x*2, )2 -> )*2
    expr = re.sub(r'(\))(\()', r'\1*\2', expr)  # )( -> )*(
    return expr

def taylor_series(f, x0, n):
    x = sp.symbols('x')
    taylor_poly = sum(f.diff(x, i).subs(x, x0) * (x - x0)**i / sp.factorial(i) for i in range(n + 1))
    return taylor_poly

def plot_taylor_series_with_function(f, x0, n, x_range=(-10, 10), y_range=(-10, 10), num_points=1000):
    x = sp.symbols('x')
    
    recognized_functions = {
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
        "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
        "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
        "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt
    }
    
    f = preprocess_expression(f)
    f_sympy = sp.sympify(f, locals=recognized_functions)
    taylor_polys = [taylor_series(f_sympy, x0, i) for i in range(1, n + 1)]
    
    x_vals = np.linspace(x_range[0], x_range[1], num_points)
    f_vals = sp.lambdify(x, f_sympy, "numpy")(x_vals)
    
    plt.figure(figsize=(12, 8))
    plt.plot(x_vals, f_vals, label=f'Original function: {f}', color='black', linewidth=2)
    
    for i, taylor_poly in enumerate(taylor_polys, 1):
        try:
            taylor_vals = sp.lambdify(x, taylor_poly, "numpy")(x_vals)
            plt.plot(x_vals, taylor_vals, label=f'Taylor n={i}')
        except Exception as e:
            print(f"Error al graficar la aproximación de Taylor n={i}: {e}")
    
    plt.legend()
    plt.title(f'Taylor Series Approximations of {sp.latex(f_sympy)} around x0={x0}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.xlim(x_range[0], x_range[1])  # Ajusta el rango del eje x
    plt.ylim(y_range[0], y_range[1])  # Ajusta el rango del eje y
    plt.grid(True)
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    
    return plot_url

def taylor_series_latex(f, x0, n):
    x = sp.symbols('x')
    recognized_functions = {
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
        "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
        "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
        "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt
    }
    
    f = preprocess_expression(f)
    f_sympy = sp.sympify(f, locals=recognized_functions)
    taylor_poly = taylor_series(f_sympy, x0, n)
    return sp.latex(taylor_poly)

@app.route('/taylor', methods=['GET', 'POST'])
def taylor():
    if request.method == 'POST':
        try:
            # Obtener datos del formulario
            function = request.form['function']
            n_terms = int(request.form['n_terms'])  # 'n_terms' es el nombre del campo en el HTML
            x_min = float(request.form['x_min'])
            x_max = float(request.form['x_max'])
            y_min = float(request.form['y_min'])
            y_max = float(request.form['y_max'])

            # Definir 'x' antes de usarlo
            x = sp.symbols('x')

            # Preparar y procesar la función
            recognized_functions = {
                "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
                "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
                "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
                "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt
            }
            
            f = sp.sympify(function, locals=recognized_functions)  # Procesa la función ingresada

            # Calcular la serie de Taylor
            def taylor_series(f, x0, n):
                return sum(f.diff(x, i).subs(x, x0) * (x - x0)**i / sp.factorial(i) for i in range(n + 1))

            taylor_polys = [taylor_series(f, 0, i) for i in range(1, n_terms + 1)]

            # Crear la gráfica
            x_vals = np.linspace(x_min, x_max, 1000)
            f_lambdified = sp.lambdify(x, f, "numpy")
            
            try:
                f_vals = f_lambdified(x_vals)
            except Exception as e:
                return f'Error al evaluar la función original: {str(e)}', 400

            plt.figure(figsize=(12, 8))
            plt.plot(x_vals, f_vals, label=f'Función Original: {function}', color='black', linewidth=2)

            for i, taylor_poly in enumerate(taylor_polys, 1):
                taylor_lambdified = sp.lambdify(x, taylor_poly, "numpy")
                
                try:
                    taylor_vals = taylor_lambdified(x_vals)
                except Exception as e:
                    return f'Error en la serie de Taylor n={i}: {str(e)}', 400

                plt.plot(x_vals, taylor_vals, label=f'Aproximación Taylor n={i}')

            plt.title(f'Serie de Taylor de {sp.latex(f)} alrededor de x0=0')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.grid(True)
            plt.legend()

            # Guardar la gráfica en un buffer de bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            graph_url = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()
            plt.close()  # Cerrar la figura para liberar memoria

            # Obtener la fórmula en LaTeX
            taylor_latex = sp.latex(taylor_series(f, 0, n_terms))

            return render_template('taylor.html', graph_url=graph_url, taylor_latex=taylor_latex)

        except Exception as e:
            # Esto asegura que siempre se devuelva una respuesta válida
            return f'Error en la ejecución de la función: {str(e)}', 400
    
    # Manejo del caso cuando se hace un GET request
    return render_template('taylor.html')



@app.route('/graficas', methods=['GET', 'POST'])
def graficas():
    if request.method == 'POST':
        try:
            # Obtener datos del formulario
            function = request.form['function']

            # Preparar y procesar la función
            x = sp.symbols('x')
            f = sp.sympify(function, evaluate=False)  # Convierte la función en una expresión SymPy

            # Crear la gráfica de la función original en un rango predeterminado
            x_min, x_max = -100, 100
            x_vals = np.linspace(x_min, x_max, 1000)
            f_lambdified = sp.lambdify(x, f, "numpy")
            
            try:
                f_vals = f_lambdified(x_vals)
            except Exception as e:
                return f'Error al evaluar la función: {str(e)}', 400

            # Verificar que todos los valores son finitos
            if not np.isfinite(f_vals).all():
                return 'Error: La función contiene valores no finitos en el rango dado.', 400

            # Crear la gráfica interactiva con Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=f_vals, mode='lines', name=f'Función: {function}'))

            # Configuración de la gráfica con zoom interactivo habilitado
            fig.update_layout(
                title=f'Gráfica de la Función: {sp.latex(f)}',
                xaxis_title='x',
                yaxis_title='f(x)',
                template="plotly_white",
                autosize=True
            )

            # Convertir la gráfica en HTML para mostrarla en la plantilla
            graph_html = pio.to_html(fig, full_html=False)

            return render_template('graficas.html', graph_html=graph_html)

        except Exception as e:
            return f'Error: {str(e)}', 400

    return render_template('graficas.html')

if __name__ == '__main__':
    app.run(debug=True)
