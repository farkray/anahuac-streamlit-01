import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Ventas de Autos",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🚗 Sistema de Predicción de Ventas de Autos</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">Powered by Pandas + Scikit-learn + Streamlit</p>', unsafe_allow_html=True)
st.markdown("---")

# Cargar datos
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/car_deal_example.csv')
        return df
    except FileNotFoundError:
        st.error("⚠️ Archivo 'car_deal_example.csv' no encontrado en la carpeta 'data/'")
        st.info("Por favor, coloca el archivo CSV en la carpeta 'data/' y recarga la aplicación.")
        st.stop()

# Preparar datos para ML
@st.cache_data
def prepare_data_for_ml(df):
    """Prepara los datos para machine learning"""
    df_ml = df.copy()
    
    # Codificar variables categóricas
    encoders = {}
    
    encoders['fuel'] = LabelEncoder()
    df_ml['fuel_encoded'] = encoders['fuel'].fit_transform(df_ml['fuel'])
    
    encoders['seller_type'] = LabelEncoder()
    df_ml['seller_type_encoded'] = encoders['seller_type'].fit_transform(df_ml['seller_type'])
    
    encoders['transmission'] = LabelEncoder()
    df_ml['transmission_encoded'] = encoders['transmission'].fit_transform(df_ml['transmission'])
    
    encoders['owner'] = LabelEncoder()
    df_ml['owner_encoded'] = encoders['owner'].fit_transform(df_ml['owner'])
    
    # Features y target
    features = ['year', 'km_driven', 'fuel_encoded', 'seller_type_encoded', 
                'transmission_encoded', 'owner_encoded']
    X = df_ml[features]
    y = df_ml['selling_price']
    
    # Dividir datos (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, features, encoders

# Sidebar - Navegación
st.sidebar.title("📊 Navegación")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Selecciona una sección:",
    ["🏠 Inicio", "📈 Análisis Exploratorio", "🤖 Predicción de Precios", "📊 Predicción de Ventas", "🎯 Clasificación de Vehículos"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.info("""
    **📚 Ciencia de Datos**
    
    Tecnologías utilizadas:
    - 🐼 Pandas
    - 🤖 Scikit-learn
    - 📊 Plotly
    - 🚀 Streamlit
""")

# Cargar datos
df = load_data()

# ========== PÁGINA: INICIO ==========
if page == "🏠 Inicio":
    st.markdown('<h2 class="sub-header">Bienvenido al Sistema de Análisis de Ventas</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total de Registros",
            value=f"{len(df):,}",
            delta="Dataset completo"
        )
    
    with col2:
        st.metric(
            label="💰 Precio Promedio",
            value=f"${df['selling_price'].mean():,.0f}",
            delta=f"±${df['selling_price'].std():,.0f}"
        )
    
    with col3:
        st.metric(
            label="🚙 Marcas Únicas",
            value=len(df['name'].str.split().str[0].unique()),
            delta="Diferentes marcas"
        )
    
    with col4:
        st.metric(
            label="📅 Rango de Años",
            value=f"{df['year'].min()} - {df['year'].max()}",
            delta=f"{df['year'].max() - df['year'].min()} años"
        )
    
    st.markdown("---")
    
    # Vista previa de datos
    st.markdown("### 📋 Vista Previa de los Datos")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Información del dataset
    st.markdown("### ℹ️ Información del Dataset")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Columnas disponibles:**")
        for col in df.columns:
            st.write(f"- `{col}` ({df[col].dtype})")
    
    with col2:
        st.markdown("**Estadísticas básicas:**")
        st.dataframe(df.describe(), use_container_width=True)

# ========== PÁGINA: ANÁLISIS EXPLORATORIO ==========
elif page == "📈 Análisis Exploratorio":
    st.markdown('<h2 class="sub-header">Análisis Exploratorio de Datos</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Distribuciones", "🔍 Correlaciones", "📈 Tendencias"])
    
    with tab1:
        st.markdown("### Distribución de Variables")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribución de precios
            fig_price = px.histogram(
                df, 
                x='selling_price', 
                nbins=50,
                title='Distribución de Precios de Venta',
                labels={'selling_price': 'Precio de Venta ($)'},
                color_discrete_sequence=['#1f77b4']
            )
            fig_price.update_layout(showlegend=False)
            st.plotly_chart(fig_price, use_container_width=True)
        
        with col2:
            # Distribución por combustible
            fuel_counts = df['fuel'].value_counts()
            fig_fuel = px.pie(
                values=fuel_counts.values,
                names=fuel_counts.index,
                title='Distribución por Tipo de Combustible',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_fuel, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Distribución por transmisión
            trans_counts = df['transmission'].value_counts()
            fig_trans = px.bar(
                x=trans_counts.index,
                y=trans_counts.values,
                title='Distribución por Tipo de Transmisión',
                labels={'x': 'Transmisión', 'y': 'Cantidad'},
                color=trans_counts.index,
                color_discrete_sequence=['#ff7f0e', '#2ca02c']
            )
            st.plotly_chart(fig_trans, use_container_width=True)
        
        with col4:
            # Distribución por año
            year_counts = df['year'].value_counts().sort_index()
            fig_year = px.line(
                x=year_counts.index,
                y=year_counts.values,
                title='Cantidad de Ventas por Año',
                labels={'x': 'Año', 'y': 'Cantidad de Ventas'},
                markers=True
            )
            st.plotly_chart(fig_year, use_container_width=True)
    
    with tab2:
        st.markdown("### Análisis de Correlaciones")
        
        # Precio vs Kilómetros
        sample_size = min(1000, len(df))
        df_sample = df.sample(sample_size)
        
        fig_km_price = px.scatter(
            df_sample,
            x='km_driven',
            y='selling_price',
            color='fuel',
            size='year',
            title='Relación entre Kilómetros Recorridos y Precio',
            labels={'km_driven': 'Kilómetros Recorridos', 'selling_price': 'Precio de Venta ($)'},
            opacity=0.6,
            hover_data=['name', 'transmission']
        )
        st.plotly_chart(fig_km_price, use_container_width=True)
        
        # Precio por año
        fig_year_price = px.box(
            df,
            x='year',
            y='selling_price',
            title='Distribución de Precios por Año del Vehículo',
            labels={'year': 'Año', 'selling_price': 'Precio de Venta ($)'}
        )
        st.plotly_chart(fig_year_price, use_container_width=True)
    
    with tab3:
        st.markdown("### Tendencias de Mercado")
        
        # Precio promedio por año
        avg_price_year = df.groupby('year')['selling_price'].mean().reset_index()
        fig_trend = px.line(
            avg_price_year,
            x='year',
            y='selling_price',
            title='Tendencia de Precio Promedio por Año',
            labels={'year': 'Año', 'selling_price': 'Precio Promedio ($)'},
            markers=True
        )
        fig_trend.update_traces(line_color='#ff7f0e', line_width=3)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Top 10 marcas más vendidas
        df['brand'] = df['name'].str.split().str[0]
        top_brands = df['brand'].value_counts().head(10)
        fig_brands = px.bar(
            x=top_brands.values,
            y=top_brands.index,
            orientation='h',
            title='Top 10 Marcas Más Vendidas',
            labels={'x': 'Cantidad de Ventas', 'y': 'Marca'},
            color=top_brands.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_brands, use_container_width=True)

# ========== PÁGINA: PREDICCIÓN DE PRECIOS ==========
elif page == "🤖 Predicción de Precios":
    st.markdown('<h2 class="sub-header">Predicción de Precios con Machine Learning</h2>', unsafe_allow_html=True)
    
    st.info("🔄 Sistema de predicción usando Scikit-learn")
    
    # Preparación de datos
    st.markdown("### 📊 Preparación de Datos")
    
    X_train, X_test, y_train, y_test, features, encoders = prepare_data_for_ml(df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📚 Datos de Entrenamiento", f"{len(X_train):,} registros", delta="80%")
    with col2:
        st.metric("🧪 Datos de Prueba", f"{len(X_test):,} registros", delta="20%")
    
    st.markdown("---")
    
    # Selección de modelo
    st.markdown("### 🎯 Selección de Modelo")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        model_type = st.selectbox(
            "Elige el algoritmo de Machine Learning:",
            ["Regresión Lineal", "Random Forest"],
            help="Regresión Lineal: Simple y rápido. Random Forest: Más preciso pero más lento."
        )
    
    with col2:
        st.markdown("**Features utilizadas:**")
        st.markdown("- Año del vehículo")
        st.markdown("- Kilómetros recorridos")
        st.markdown("- Tipo de combustible")
        st.markdown("- Tipo de vendedor")
        st.markdown("- Transmisión")
        st.markdown("- Tipo de propietario")
    
    if st.button("🚀 Entrenar Modelo", type="primary", use_container_width=True):
        with st.spinner("Entrenando modelo de Machine Learning..."):
            
            # Crear y entrenar modelo
            if model_type == "Regresión Lineal":
                model = LinearRegression()
                model_name = "Linear Regression"
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                model_name = "Random Forest"
            
            # Entrenar
            model.fit(X_train, y_train)
            
            # Predicciones
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Evaluación
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            mae_test = mean_absolute_error(y_test, y_pred_test)
            
            st.success(f"✅ Modelo {model_name} entrenado exitosamente!")
            
            # Métricas
            st.markdown("### 📈 Métricas del Modelo")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("RMSE (Test)", f"${rmse_test:,.0f}", 
                         delta=f"{((rmse_test-rmse_train)/rmse_train*100):.1f}%",
                         help="Root Mean Square Error - Menor es mejor")
            
            with col2:
                st.metric("R² Score (Test)", f"{r2_test:.4f}",
                         delta=f"{((r2_test-r2_train)*100):.1f}%",
                         help="Coeficiente de determinación - Más cercano a 1 es mejor")
            
            with col3:
                accuracy = r2_test * 100
                st.metric("Precisión", f"{accuracy:.2f}%",
                         help="Porcentaje de varianza explicada")
            
            with col4:
                st.metric("MAE (Test)", f"${mae_test:,.0f}",
                         help="Mean Absolute Error - Error promedio absoluto")
            
            # Comparación Train vs Test
            st.markdown("### 📊 Comparación: Entrenamiento vs Prueba")
            comparison_df = pd.DataFrame({
                'Métrica': ['RMSE', 'R² Score', 'MAE'],
                'Entrenamiento': [f"${rmse_train:,.0f}", f"{r2_train:.4f}", f"${mean_absolute_error(y_train, y_pred_train):,.0f}"],
                'Prueba': [f"${rmse_test:,.0f}", f"{r2_test:.4f}", f"${mae_test:,.0f}"]
            })
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Visualización de predicciones
            st.markdown("### 🎯 Visualización de Predicciones")
            
            # Crear DataFrame con predicciones
            predictions_df = pd.DataFrame({
                'Real': y_test,
                'Predicho': y_pred_test
            })
            
            fig_pred = go.Figure()
            
            # Puntos de predicción
            fig_pred.add_trace(go.Scatter(
                x=predictions_df['Real'],
                y=predictions_df['Predicho'],
                mode='markers',
                marker=dict(
                    size=6,
                    color=predictions_df['Real'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Precio Real ($)"),
                    opacity=0.6
                ),
                name='Predicciones',
                hovertemplate='<b>Real:</b> $%{x:,.0f}<br><b>Predicho:</b> $%{y:,.0f}<extra></extra>'
            ))
            
            # Línea diagonal perfecta
            max_val = max(predictions_df['Real'].max(), predictions_df['Predicho'].max())
            min_val = min(predictions_df['Real'].min(), predictions_df['Predicho'].min())
            
            fig_pred.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='Predicción Perfecta'
            ))
            
            fig_pred.update_layout(
                title='Valores Reales vs Predicciones',
                xaxis_title='Precio Real ($)',
                yaxis_title='Precio Predicho ($)',
                hovermode='closest',
                height=500
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Importancia de features (solo para Random Forest)
            if model_type == "Random Forest":
                st.markdown("### 🎯 Importancia de Variables")
                
                feature_importance = pd.DataFrame({
                    'Feature': ['Año', 'Kilómetros', 'Combustible', 'Vendedor', 'Transmisión', 'Propietario'],
                    'Importancia': model.feature_importances_
                }).sort_values('Importancia', ascending=False)
                
                fig_importance = px.bar(
                    feature_importance,
                    x='Importancia',
                    y='Feature',
                    orientation='h',
                    title='Importancia de Cada Variable en el Modelo',
                    color='Importancia',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_importance, use_container_width=True)

# ========== PÁGINA: PREDICCIÓN DE VENTAS ==========
elif page == "📊 Predicción de Ventas":
    st.markdown('<h2 class="sub-header">Predicción de Ventas con Machine Learning</h2>', unsafe_allow_html=True)
    
    st.info("📈 Modelo de predicción basado en series temporales")
    
    # Agregar datos por año del modelo
    sales_by_year = df.groupby('year').size().reset_index(name='cantidad')
    sales_by_year = sales_by_year.sort_values('year')
    
    # Verificar que hay suficientes datos
    if len(sales_by_year) < 3:
        st.error("⚠️ Se necesitan al menos 3 años de datos para hacer predicciones")
        st.stop()
    
    # Mostrar histórico
    st.markdown("### 📊 Datos Históricos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📈 Total Autos", f"{len(df):,}")
    
    with col2:
        promedio = int(sales_by_year['cantidad'].mean())
        st.metric("📊 Promedio/Año", f"{promedio:,}")
    
    with col3:
        ultimo_anio = int(sales_by_year['year'].max())
        ultimo_valor = int(sales_by_year[sales_by_year['year'] == ultimo_anio]['cantidad'].values[0])
        st.metric(f"🔹 Año {ultimo_anio}", f"{ultimo_valor:,}")
    
    # Gráfico histórico
    fig_hist = px.bar(
        sales_by_year,
        x='year',
        y='cantidad',
        title='Cantidad de Autos por Año del Modelo',
        labels={'year': 'Año', 'cantidad': 'Cantidad'}
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")
    
    # Configuración de predicción
    st.markdown("### 🤖 Modelo de Predicción ML")
    
    col1, col2 = st.columns(2)
    
    with col1:
        years_predict = st.slider("Años a predecir:", 1, 3, 1)
        
    with col2:
        modelo_ml = st.selectbox(
            "Algoritmo:",
            ["Random Forest", "Gradient Boosting", "Regresión Polinomial"],
            help="Modelos de Machine Learning avanzados"
        )
    
    if st.button("🚀 Entrenar y Predecir", type="primary", use_container_width=True):
        with st.spinner("Entrenando modelo de ML..."):
            
            # Preparar features avanzados
            sales_by_year['year_norm'] = (sales_by_year['year'] - sales_by_year['year'].min()) / (sales_by_year['year'].max() - sales_by_year['year'].min())
            sales_by_year['year_squared'] = sales_by_year['year'] ** 2
            
            # Media móvil como feature
            sales_by_year['moving_avg'] = sales_by_year['cantidad'].rolling(window=2, min_periods=1).mean()
            
            # Diferencia año anterior
            sales_by_year['diff'] = sales_by_year['cantidad'].diff().fillna(0)
            
            # Preparar datos de entrenamiento
            X = sales_by_year[['year', 'year_squared', 'moving_avg', 'diff']].values
            y = sales_by_year['cantidad'].values
            
            # Entrenar modelo
            if modelo_ml == "Random Forest":
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            elif modelo_ml == "Gradient Boosting":
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
            else:  # Regresión Polinomial
                from sklearn.preprocessing import PolynomialFeatures
                from sklearn.linear_model import Ridge
                from sklearn.pipeline import make_pipeline
                model = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=1.0))
                X = sales_by_year[['year']].values
            
            model.fit(X, y)
            
            # Generar predicciones futuras
            future_years = np.arange(ultimo_anio + 1, ultimo_anio + years_predict + 1)
            
            if modelo_ml in ["Random Forest", "Gradient Boosting"]:
                # Crear features para años futuros
                last_moving_avg = sales_by_year['moving_avg'].iloc[-1]
                last_diff = sales_by_year['diff'].iloc[-1]
                
                future_features = []
                for year in future_years:
                    year_norm = (year - sales_by_year['year'].min()) / (sales_by_year['year'].max() - sales_by_year['year'].min())
                    year_sq = year ** 2
                    future_features.append([year, year_sq, last_moving_avg, last_diff])
                
                X_future = np.array(future_features)
            else:
                X_future = future_years.reshape(-1, 1)
            
            predictions = model.predict(X_future)
            predictions = np.maximum(predictions, 0).astype(int)
            
            # Ajustar predicciones para que sean más realistas (evitar caídas drásticas)
            # Si la predicción es menos del 50% del último valor, ajustar
            for i in range(len(predictions)):
                if predictions[i] < ultimo_valor * 0.5:
                    # Usar tendencia más conservadora
                    tendencia = (ultimo_valor - sales_by_year['cantidad'].iloc[-2]) if len(sales_by_year) > 1 else 0
                    predictions[i] = int(ultimo_valor + tendencia * (i + 1))
            
            # Crear DataFrame de resultados
            pred_df = pd.DataFrame({
                'Año': future_years,
                'Predicción': predictions
            })
            
            st.success("✅ Modelo entrenado correctamente")
            
            # Métricas del modelo
            st.markdown("### 📊 Evaluación del Modelo")
            
            y_pred_train = model.predict(X)
            y_pred_train = np.maximum(y_pred_train, 0)
            
            from sklearn.metrics import r2_score, mean_absolute_error
            r2 = r2_score(y, y_pred_train)
            mae = mean_absolute_error(y, y_pred_train)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{r2:.3f}", help="Calidad del ajuste (0-1)")
            with col2:
                st.metric("MAE", f"{mae:.0f}", help="Error absoluto promedio")
            with col3:
                st.metric("Modelo", modelo_ml)
            
            # Resultados
            st.markdown("### 📈 Predicciones")
            
            cols = st.columns(len(pred_df))
            for col, (_, row) in zip(cols, pred_df.iterrows()):
                with col:
                    cambio = int(row['Predicción']) - ultimo_valor
                    pct = (cambio / ultimo_valor * 100) if ultimo_valor > 0 else 0
                    st.metric(
                        f"Año {int(row['Año'])}",
                        f"{int(row['Predicción']):,}",
                        f"{pct:+.1f}%"
                    )
            
            # Tabla
            pred_df['Cambio_vs_Base'] = pred_df['Predicción'] - ultimo_valor
            pred_df['Cambio_%'] = ((pred_df['Predicción'] - ultimo_valor) / ultimo_valor * 100).round(1)
            st.dataframe(pred_df, use_container_width=True, hide_index=True)
            
            # Gráfico
            st.markdown("### 📉 Visualización")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=sales_by_year['year'],
                y=sales_by_year['cantidad'],
                mode='lines+markers',
                name='Histórico',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ))
            
            fig.add_trace(go.Scatter(
                x=pred_df['Año'],
                y=pred_df['Predicción'],
                mode='lines+markers',
                name='Predicción ML',
                line=dict(color='red', width=3, dash='dash'),
                marker=dict(size=10, symbol='star')
            ))
            
            fig.update_layout(
                title=f'Proyección con {modelo_ml}',
                xaxis_title='Año',
                yaxis_title='Cantidad de Autos',
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Análisis
            st.markdown("### 💡 Análisis")
            
            tendencia = "crecimiento" if predictions[-1] > ultimo_valor else "decrecimiento"
            cambio_total = ((predictions[-1] - ultimo_valor) / ultimo_valor * 100)
            
            if abs(cambio_total) < 5:
                st.info(f"📊 Tendencia estable: {cambio_total:+.1f}% hacia {int(future_years[-1])}")
            elif cambio_total > 0:
                st.success(f"📈 {tendencia.capitalize()}: {cambio_total:+.1f}% hacia {int(future_years[-1])}")
            else:
                st.warning(f"📉 {tendencia.capitalize()}: {cambio_total:+.1f}% hacia {int(future_years[-1])}")
    
    st.markdown("---")
    
    # Análisis por categorías
    st.markdown("### 🔍 Análisis por Segmento")
    
    tab1, tab2 = st.tabs(["Por Combustible", "Por Marca"])
    
    with tab1:
        fuel_year = df.groupby(['year', 'fuel']).size().reset_index(name='cantidad')
        
        fig = px.line(
            fuel_year,
            x='year',
            y='cantidad',
            color='fuel',
            title='Evolución por Tipo de Combustible',
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribución actual
        fuel_dist = df['fuel'].value_counts()
        st.markdown("**Distribución Actual:**")
        for fuel, count in fuel_dist.items():
            pct = (count / len(df)) * 100
            st.write(f"**{fuel}:** {count:,} ({pct:.1f}%)")
    
    with tab2:
        df['brand'] = df['name'].str.split().str[0]
        top_brands = df['brand'].value_counts().head(10)
        
        fig = px.bar(
            x=top_brands.index,
            y=top_brands.values,
            title='Top 10 Marcas',
            labels={'x': 'Marca', 'y': 'Cantidad'}
        )
        st.plotly_chart(fig, use_container_width=True)

# ========== PÁGINA: CLASIFICACIÓN ==========
elif page == "🎯 Clasificación de Vehículos":
    st.markdown('<h2 class="sub-header">Clasificación de Tipos de Vehículos</h2>', unsafe_allow_html=True)
    
    st.info("🔍 Análisis de clasificación basado en características del vehículo")
    
    # Extraer marca del nombre
    df['brand'] = df['name'].str.split().str[0]
    
    # Crear categorías de precio
    df['price_category'] = pd.cut(
        df['selling_price'],
        bins=[0, 200000, 500000, 1000000, float('inf')],
        labels=['Económico', 'Medio', 'Premium', 'Lujo']
    )
    
    # Visualizaciones de clasificación
    tab1, tab2, tab3 = st.tabs(["🏷️ Por Precio", "⚙️ Por Características", "🔍 Segmentación"])
    
    with tab1:
        st.markdown("### Clasificación por Rango de Precio")
        
        col1, col2 = st.columns(2)
        
        with col1:
            price_dist = df['price_category'].value_counts()
            fig_price_cat = px.pie(
                values=price_dist.values,
                names=price_dist.index,
                title='Distribución por Categoría de Precio',
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.3
            )
            st.plotly_chart(fig_price_cat, use_container_width=True)
        
        with col2:
            # Gráfico de barras con promedio por categoría
            avg_by_cat = df.groupby('price_category')['selling_price'].mean().reset_index()
            fig_avg = px.bar(
                avg_by_cat,
                x='price_category',
                y='selling_price',
                title='Precio Promedio por Categoría',
                labels={'price_category': 'Categoría', 'selling_price': 'Precio Promedio ($)'},
                color='selling_price',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_avg, use_container_width=True)
        
        # Tabla de resumen
        st.markdown("#### 📊 Resumen Detallado por Categoría")
        summary = df.groupby('price_category').agg({
            'selling_price': ['mean', 'min', 'max', 'count']
        }).round(0)
        summary.columns = ['Promedio ($)', 'Mínimo ($)', 'Máximo ($)', 'Cantidad']
        st.dataframe(summary, use_container_width=True)
    
    with tab2:
        st.markdown("### Clasificación por Características Técnicas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Por tipo de combustible y transmisión
            fuel_trans = df.groupby(['fuel', 'transmission']).size().reset_index(name='count')
            fig_fuel_trans = px.sunburst(
                fuel_trans,
                path=['fuel', 'transmission'],
                values='count',
                title='Jerarquía: Combustible → Transmisión',
                color='count',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_fuel_trans, use_container_width=True)
        
        with col2:
            # Por propietario
            owner_dist = df['owner'].value_counts()
            fig_owner = px.bar(
                x=owner_dist.index,
                y=owner_dist.values,
                title='Distribución por Tipo de Propietario',
                labels={'x': 'Tipo de Propietario', 'y': 'Cantidad'},
                color=owner_dist.values,
                color_continuous_scale='Teal'
            )
            st.plotly_chart(fig_owner, use_container_width=True)
    
    with tab3:
        st.markdown("### Segmentación Avanzada de Mercado")
        
        # Segmentación por marca y precio
        brand_stats = df.groupby('brand').agg({
            'selling_price': ['mean', 'count'],
            'km_driven': 'mean'
        }).reset_index()
        brand_stats.columns = ['brand', 'avg_price', 'total_sales', 'avg_km']
        brand_stats = brand_stats.sort_values('total_sales', ascending=False).head(15)
        
        fig_segment = px.scatter(
            brand_stats,
            x='total_sales',
            y='avg_price',
            size='total_sales',
            color='brand',
            text='brand',
            title='Segmentación: Volumen vs Precio Promedio (Top 15 Marcas)',
            labels={'total_sales': 'Volumen de Ventas', 'avg_price': 'Precio Promedio ($)'},
            size_max=60
        )
        fig_segment.update_traces(textposition='top center', textfont_size=10)
        fig_segment.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig_segment, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🎓 Sistema de Predicción de Ventas de Autos | Ciencia de Datos</p>
        <p>Desarrollado con <b>Streamlit</b> + <b>Pandas</b> + <b>Scikit-learn</b> + <b>Plotly</b></p>
    </div>
    """,
    unsafe_allow_html=True
)