# ============================================
# HELLO.PY - Introducción a Streamlit
# ============================================
# Este archivo es un ejemplo educativo básico
# para aprender los fundamentos de Streamlit

import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ============================================
# 1. CONFIGURACIÓN BÁSICA DE LA PÁGINA
# ============================================
# set_page_config() debe ser lo primero que ejecutes
st.set_page_config(
    page_title="Hello Streamlit",  # Título en la pestaña del navegador
    page_icon="👋",              # Emoji que aparece en la pestaña
    layout="wide"                 # Layout: "centered" o "wide"
)

# ============================================
# 2. TÍTULO Y TEXTO BÁSICO
# ============================================
# Hay varias formas de mostrar texto:

# Título principal (más grande)
st.title("👋 ¡Hola! Bienvenido a Streamlit")

# Subtítulo
st.subheader("Introducción a Streamlit con WordCloud")

# Texto normal
st.write("Esta es una aplicación de ejemplo para aprender los conceptos básicos.")

# Línea divisoria
st.markdown("---")

# ============================================
# 3. INPUT DEL USUARIO - NOMBRE
# ============================================
# text_input() crea una caja de texto donde el usuario puede escribir
nombre = st.text_input(
    "¿Cómo te llamas?",                  # Label que se muestra
    placeholder="Escribe tu nombre aquí",  # Texto de ayuda
    help="Introduce tu nombre para personalizar la experiencia"
)

# Verificar si el usuario escribió algo
if nombre:
    # Si hay nombre, mostramos un saludo personalizado
    st.success(f"¡Hola {nombre}! 🎉 Bienvenido a tu primera app en Streamlit")
else:
    # Si no hay nombre, mostramos un mensaje genérico
    st.info("👆 Por favor, introduce tu nombre arriba")

st.markdown("---")

# ============================================
# 4. MENÚ DE NAVEGACIÓN
# ============================================
# Usamos sidebar (barra lateral) para crear un menú
st.sidebar.title("📊 Menú de Navegación")
st.sidebar.markdown("Selecciona una opción:")

# Radio button para crear un menú con opciones
opcion = st.sidebar.radio(
    "Elige una sección:",
    ["🏠 Inicio", "📊 WordCloud Demo", "ℹ️ Ayuda"],
    label_visibility="collapsed"  # Ocultar el label
)

# Información adicional en el sidebar
st.sidebar.markdown("---")
st.sidebar.info("""
    **📚 Conceptos aprendidos:**
    - Configuración de página
    - Inputs de usuario
    - Menú de navegación
    - Carga de datos
    - Visualización con WordCloud
""")

# ============================================
# 5. PÁGINAS / SECCIONES
# ============================================
# Dependiendo de la opción seleccionada, mostramos diferente contenido

# ---------- PÁGINA: INICIO ----------
if opcion == "🏠 Inicio":
    st.header("🏠 Página de Inicio")
    
    st.markdown("""
    ### ¿Qué es Streamlit?
    
    **Streamlit** es un framework de Python que permite crear aplicaciones web
    interactivas de forma rápida y sencilla, ideal para:
    
    - 📊 Visualización de datos
    - 🤖 Machine Learning
    - 📈 Dashboards
    - 🎓 Educación
    
    ### Componentes básicos que usamos aquí:
    
    1. **st.title()** - Títulos grandes
    2. **st.text_input()** - Caja de texto para input
    3. **st.sidebar** - Barra lateral con menú
    4. **st.dataframe()** - Mostrar tablas
    5. **st.pyplot()** - Mostrar gráficos
    """)
    
    # Mostrar ejemplo de código
    st.code("""
    # Ejemplo simple:
    import streamlit as st
    
    nombre = st.text_input("Tu nombre:")
    st.write(f"Hola {nombre}!")
    """, language="python")

# ---------- PÁGINA: WORDCLOUD DEMO ----------
elif opcion == "📊 WordCloud Demo":
    st.header("📊 WordCloud de Keywords")
    
    st.markdown("""
    En esta sección vamos a:
    1. Cargar datos desde un CSV
    2. Filtrar las top 150 keywords por volumen de búsqueda
    3. Generar un WordCloud visual
    """)
    
    # ============================================
    # 6. CARGAR DATOS DESDE CSV
    # ============================================
    # Intentar cargar el archivo CSV
    try:
        # pd.read_csv() lee archivos CSV y los convierte en DataFrame
        df = pd.read_csv('data/data_demo.csv')
        
        # MODIFICADO: Definir las columnas que usaremos
        # Revisa que estos nombres coincidan EXACTAMENTE con tu CSV (mayúsculas/minúsculas)
        COLUMNA_KEYWORDS = 'Keyword'
        COLUMNA_VOLUMEN = 'Search Volume'

        # MODIFICADO: Verificar si las columnas esperadas existen
        if COLUMNA_KEYWORDS not in df.columns or COLUMNA_VOLUMEN not in df.columns:
            st.error(f"❌ Error: El archivo CSV no contiene las columnas requeridas.")
            st.info(f"""
            Asegúrate de que tu archivo CSV tenga una columna llamada: `{COLUMNA_KEYWORDS}`
            Y otra columna llamada: `{COLUMNA_VOLUMEN}`
            """)
            st.markdown("**Columnas encontradas en tu archivo:**")
            st.code(f"[{', '.join(df.columns)}]")
        
        else:
            st.success(f"✅ Datos cargados: {len(df)} registros encontrados")
            
            # Mostrar información básica del dataset
            st.markdown("### 📋 Vista Previa de los Datos")
            
            # Crear columnas para organizar la información
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Keywords", len(df))
            
            with col2:
                # MODIFICADO: Calculamos el volumen total de búsqueda
                volumen_total = df[COLUMNA_VOLUMEN].sum()
                st.metric("Volumen Total", f"{volumen_total:,}")
            
            # Mostrar las primeras filas del dataset
            st.markdown("**Primeros 10 registros:**")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown("---")
            
            # ============================================
            # 7. FILTRAR Y PROCESAR DATOS
            # ============================================
            st.markdown("### 🔍 Filtrado de Datos")
            
            # MODIFICADO: Usar la columna de volumen para ordenar
            st.write(f"Filtrando por la columna: **{COLUMNA_VOLUMEN}**")
            
            # Ordenar por Search volume de mayor a menor
            df_sorted = df.sort_values(COLUMNA_VOLUMEN, ascending=False)
            
            # Tomar solo las top 150
            top_150 = df_sorted.head(150)
            
            st.info(f"📊 Seleccionadas las top 150 keywords con mayor volumen de búsqueda")
            
            # Mostrar estadísticas de las top 150
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top 5 Keywords (por volumen):**")
                # MODIFICADO: Iterar sobre el DataFrame de frecuencias
                for idx, row in top_150.head(5).iterrows():
                    # MODIFICADO: Usar las nuevas columnas 'Keyword' y 'Frecuencia'
                    st.write(f"{idx+1}. **{row[COLUMNA_KEYWORDS]}** - {row[COLUMNA_VOLUMEN]:,} búsquedas")
            
            with col2:
                # MODIFICADO: Calcular métricas basadas en Volumen
                volumen_top150 = top_150[COLUMNA_VOLUMEN].sum()
                
                if volumen_total > 0:
                    porcentaje = (volumen_top150 / volumen_total) * 100
                else:
                    porcentaje = 0
                
                st.metric(
                    "Volumen Top 150",
                    f"{volumen_top150:,}",
                    f"{porcentaje:.1f}% del total"
                )
            
            st.markdown("---")
            
            # ============================================
            # 8. GENERAR WORDCLOUD
            # ============================================
            st.markdown("### ☁️ WordCloud Generado")
            
            st.markdown("""
            El **WordCloud** visualiza las keywords donde:
            - Palabras más **grandes** = Mayor volumen de búsqueda
            - Palabras más **pequeñas** = Menor volumen de búsqueda
            """)
            
            # Botón para generar el WordCloud
            if st.button("🎨 Generar WordCloud", type="primary"):
                with st.spinner("Generando WordCloud..."):
                    
                    # Crear diccionario: keyword -> search volume
                    # El tamaño de cada palabra dependerá de su volumen
                    word_freq = {}
                    for _, row in top_150.iterrows():
                        # MODIFICADO: Usamos 'Keyword' y 'Search volume'
                        word_freq[row[COLUMNA_KEYWORDS]] = row[COLUMNA_VOLUMEN]
                    
                    # Crear el WordCloud
                    wordcloud = WordCloud(
                        width=1200,                # Ancho en píxeles
                        height=600,                # Alto en píxeles
                        background_color='white',  # Color de fondo
                        colormap='plasma',         # MODIFICADO: Color de azul/morado a naranja/amarillo
                        relative_scaling=0.5,      # Escala relativa de tamaños
                        min_font_size=10           # Tamaño mínimo de fuente
                    ).generate_from_frequencies(word_freq)
                    
                    # Crear figura de matplotlib para mostrar el WordCloud
                    fig, ax = plt.subplots(figsize=(15, 8))
                    
                    # Mostrar el wordcloud
                    ax.imshow(wordcloud, interpolation='bilinear')
                    
                    # Quitar ejes (no necesitamos coordenadas)
                    ax.axis('off')
                    
                    # Agregar título
                    ax.set_title(
                        f'Top 150 Keywords por Volumen de Búsqueda\n(Generado por {nombre if nombre else "Usuario"})',
                        fontsize=16,
                        pad=20
                    )
                    
                    # Mostrar en Streamlit
                    st.pyplot(fig)
                    
                    st.success("✅ WordCloud generado exitosamente!")
                    
                    # Información adicional
                    st.info("""
                    💡 **Tip:** Las palabras más grandes representan las keywords 
                    con mayor volumen de búsqueda. Este tipo de visualización 
                    es útil para identificar rápidamente tendencias y términos importantes.
                    """)
            
            # ============================================
            # 9. DESCARGAR DATOS PROCESADOS
            # ============================================
            st.markdown("---")
            st.markdown("### 💾 Exportar Datos")
            
            # Convertir DataFrame (de frecuencias) a CSV para descarga
            csv = top_150.to_csv(index=False).encode('utf-8')
            
            # Botón de descarga
            st.download_button(
                label="📥 Descargar Top 150 Keywords (CSV)",
                data=csv,
                file_name="top_150_keywords_frecuencia.csv",
                mime="text/csv",
                help="Descargar las 150 keywords más frecuentes con su conteo"
            )
    
    except FileNotFoundError:
        # Si no se encuentra el archivo, mostrar error
        st.error("❌ Error: No se encontró el archivo 'data/data_demo.csv'")
        st.info("""
        **Para usar esta demo:**
        1. Crea una carpeta llamada `data/` (en la misma ubicación que tu script)
        2. Coloca tu archivo `data_demo.csv` dentro
        3. MODIFICADO: El archivo debe tener las columnas: `Keyword` y `Search Volume`
        """)
        
        # Mostrar ejemplo de cómo debería verse el CSV
        st.markdown("**Ejemplo de estructura del CSV:**")
        # MODIFICADO: Ejemplo de CSV actualizado
        st.code("""
Keyword,Search Volume
python tutorial,10000
machine learning,8500
data science,7200
streamlit app,5600
...
        """)
    
    except Exception as e:
        # Capturar otros posibles errores (ej. permisos)
        st.error(f"❌ Ocurrió un error inesperado al leer el archivo:")
        st.exception(e)

# ---------- PÁGINA: AYUDA ----------
elif opcion == "ℹ️ Ayuda":
    st.header("ℹ️ Ayuda y Documentación")
    
    st.markdown("""
    ### 🚀 Cómo ejecutar esta aplicación
    
    1. **Instalar dependencias:**
    ```bash
    pip install streamlit pandas wordcloud matplotlib
    ```
    
    2. **Ejecutar la aplicación:**
    ```bash
    streamlit run hello.py
    ```
    
    3. **Preparar los datos:**
    - Crear carpeta `data/`
    - Colocar archivo `data_demo.csv` con columnas: `Keyword`, `Search Volume`
    
    ---
    
    ### 📚 Conceptos de Streamlit usados
    
    | Componente | Descripción |
    |------------|-------------|
    | `st.title()` | Título principal |
    | `st.header()` | Encabezado de sección |
    | `st.text_input()` | Input de texto |
    | `st.button()` | Botón clickeable |
    | `st.sidebar` | Barra lateral |
    | `st.dataframe()` | Tabla de datos |
    | `st.pyplot()` | Gráfico matplotlib |
    | `st.metric()` | Métrica con valor y delta |
    | `st.success()` | Mensaje de éxito |
    | `st.error()` | Mensaje de error |
    | `st.info()` | Mensaje informativo |
    
    ---
    
    ### 🎨 WordCloud - Librería
    
    **WordCloud** es una librería de Python para crear nubes de palabras.
    
    **Instalación:**
    ```bash
    pip install wordcloud
    ```
    
    **Uso básico:**
    ```python
    from wordcloud import WordCloud
    
    # Crear diccionario palabra: frecuencia
    words = {"python": 100, "streamlit": 80, "data": 60}
    
    # Generar WordCloud
    wc = WordCloud().generate_from_frequencies(words)
    
    # Mostrar
    plt.imshow(wc)
    plt.show()
    ```
    
    ---
    
    ### 🔗 Enlaces Útiles
    
    - [Documentación de Streamlit](https://docs.streamlit.io/)
    - [Documentación de WordCloud](https://amueller.github.io/word_cloud/)
    - [Pandas Documentation](https://pandas.pydata.org/docs/)
    - [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
    """)
    
    # Sección de contacto/créditos
    st.markdown("---")
    st.info("""
    **👨‍💻 Desarrollado para:**
    - Clase de Ciencia de Datos
    - Introducción a Streamlit
    - Visualización de datos
    """)

# ============================================
# 10. FOOTER
# ============================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📚 Hello Streamlit - Aplicación de Ejemplo</p>
        <p style='font-size: 0.8em;'>Introducción a Streamlit con WordCloud</p>
    </div>
""", unsafe_allow_html=True)

# ============================================
# NOTAS FINALES PARA EL ESTUDIANTE
# ============================================
"""
🎓 CONCEPTOS APRENDIDOS EN ESTE ARCHIVO:

1. Configuración básica de Streamlit
2. Componentes de texto (title, header, write, markdown)
3. Inputs del usuario (text_input)
4. Navegación con sidebar y radio buttons
5. Carga de datos con Pandas
6. Procesamiento de datos (value_counts, head)
7. Métricas y columnas para organizar información
8. Generación de visualizaciones (WordCloud + Matplotlib)
9. Manejo de errores (try/except)
10. Botones de descarga

💡 EJERCICIOS SUGERIDOS:

1. Agrega más opciones al menú
2. Cambia los colores del WordCloud (colormap)
3. Permite al usuario elegir cuántas keywords ver (st.slider)
4. Agrega más gráficos (un st.bar_chart con el top 10)
5. Agrega filtros (por ejemplo, por 'Keyword Intents' si esa columna existe)

🚀 PRÓXIMOS PASOS:

- Aprende sobre st.cache_data para optimizar carga de datos
- Explora diferentes tipos de gráficos con Plotly
- Implementa machine learning en Streamlit
- Crea dashboards más complejos
"""