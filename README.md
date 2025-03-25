# 📦 Guía para Crear una Cuenta en PyPI y Subir una Librería

## 1️⃣ Crear una Cuenta en PyPI

1. **Ir a la página de registro**
   - Abre tu navegador y ve a [https://pypi.org/account/register/](https://pypi.org/account/register/)
   
2. **Completar el formulario de registro**
   - Nombre de usuario (único)
   - Correo electrónico válido
   - Contraseña segura

3. **Confirmar tu correo**
   - PyPI enviará un correo de verificación
   - Haz clic en el enlace para confirmar tu cuenta

4. **(Opcional) Configurar autenticación en dos pasos (2FA)**
   - Ve a tu perfil en [https://pypi.org/manage/account/](https://pypi.org/manage/account/)
   - Usa Google Authenticator, Authy o Microsoft Authenticator para mayor seguridad

---

## 2️⃣ Preparar el Proyecto

1. **Estructura del proyecto**
   ```plaintext
   mi_libreria/
   ├── mi_libreria/
   │   ├── __init__.py
   │   ├── modulo1.py
   │   ├── modulo2.py
   ├── setup.py
   ├── README.md
   ├── LICENSE
   ├── requirements.txt
   ```
   
2. **Escribir `setup.py`**
   ```code
   from setuptools import setup, find_packages

   setup(
       name="mi_libreria",
       version="0.1.0",
       packages=find_packages(),
       install_requires=["numpy", "pandas"],  # Dependencias
       author="Tu Nombre",
       author_email="tuemail@example.com",
       description="Descripción breve de tu librería",
       long_description=open("README.md").read(),
       long_description_content_type="text/markdown",
       url="https://github.com/tuusuario/mi_libreria",  # URL del repositorio
       classifiers=[
           "Programming Language :: Python :: 3",
           "License :: OSI Approved :: MIT License",
           "Operating System :: OS Independent",
       ],
       python_requires='>=3.6',
   )
   ```

3. **Crear un archivo `__init__.py`** dentro de `mi_libreria/` para definirlo como paquete.
   ```code
   __version__ = "0.1.0"
   ```

4. **(Opcional) Crear `requirements.txt`** con las dependencias:
   ```plaintext
   numpy
   pandas
   ```

---

## 3️⃣ Generar los Archivos para la Distribución

1. **Instalar herramientas necesarias**
   ```code
   pip install --upgrade setuptools wheel twine
   ```

2. **Generar el paquete**
   ```code
   python setup.py sdist bdist_wheel
   ```
   Esto generará un directorio `dist/` con los archivos listos para subir.

---

## 4️⃣ Subir el Paquete a PyPI

### **Método 1: Usando usuario y contraseña**
```bash
   twine upload dist/*
```

### **Método 2: Usando API Token (recomendado)**

1. **Generar un Token de API**
   - Ve a [https://pypi.org/manage/account/](https://pypi.org/manage/account/)
   - Haz clic en "Add API Token"
   - Guarda el token generado de forma segura

2. **Subir con el token**
   ```bash
   twine upload --username __token__ --password TU_TOKEN dist/*
   ```

---

## 5️⃣ Instalar y Probar la Librería

1. **Instalar desde PyPI**
   ```bash
   pip install mi_libreria
   ```

2. **Probar importación**
   ```code
   import mi_libreria
   print(mi_libreria.__version__)
   ```

---

## ✅ ¡Listo! Tu librería está en PyPI 🎉


