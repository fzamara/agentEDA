

# main.py
import streamlit as st
import pandas as pd
import numpy as np
import logging
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
from langchain.prompts import ChatPromptTemplate
import os
# LangChain imports
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage
from langchain.tools import BaseTool
from langchain.agents import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory.chat_memory import BaseChatMemory
from typing import List, Any, Dict,Optional
from langchain_openai import AzureChatOpenAI
import json
from dotenv import load_dotenv
from langchain.chains import LLMChain
load_dotenv() # Carrega variáveis de ambiente do arquivo .env
logger = logging.getLogger(__name__)

# Configuração da página
st.set_page_config(
    page_title="Especialista em Análise de Dados",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DataAnalysisMemory:
    """Memória personalizada para armazenar análises e conclusões"""
    
    def __init__(self):
        self.analyses = []
        self.conclusions = []
        self.dataset_info = {}
    
    def add_analysis(self, analysis_type: str, result: Any, visualization: str = None):
        """Adiciona uma análise à memória"""
        analysis = {
            "type": analysis_type,
            "result": result,
            "visualization": visualization,
            "timestamp": pd.Timestamp.now()
        }
        self.analyses.append(analysis)
    
    def add_conclusion(self, conclusion: str):
        """Adiciona uma conclusão à memória"""
        self.conclusions.append({
            "conclusion": conclusion,
            "timestamp": pd.Timestamp.now()
        })
    
    def get_memory_summary(self) -> str:
        """Retorna um resumo da memória para o agente"""
        summary = f"Dataset Info: {self.dataset_info}\n"
        summary += f"Total de análises realizadas: {len(self.analyses)}\n"
        summary += "Análises recentes:\n"
        for analysis in self.analyses[-5:]:
            summary += f"- {analysis['type']}: {str(analysis['result'])[:100]}...\n"
        summary += "Conclusões:\n"
        for conclusion in self.conclusions:
            summary += f"- {conclusion['conclusion']}\n"
        return summary    
    def __init__(self):
        super().__init__()
        self.analyses = []
        self.conclusions = []
        self.dataset_info = {}
    
    def add_analysis(self, analysis_type: str, result: Any, visualization: str = None):
        """Adiciona uma análise à memória"""
        analysis = {
            "type": analysis_type,
            "result": result,
            "visualization": visualization,
            "timestamp": pd.Timestamp.now()
        }
        self.analyses.append(analysis)
    
    def add_conclusion(self, conclusion: str):
        """Adiciona uma conclusão à memória"""
        self.conclusions.append({
            "conclusion": conclusion,
            "timestamp": pd.Timestamp.now()
        })
    
    def get_memory_summary(self) -> str:
        """Retorna um resumo da memória para o agente"""
        summary = f"Dataset Info: {self.dataset_info}\n"
        summary += f"Total de análises realizadas: {len(self.analyses)}\n"
        summary += "Análises recentes:\n"
        for analysis in self.analyses[-5:]:
            summary += f"- {analysis['type']}: {str(analysis['result'])[:100]}...\n"
        summary += "Conclusões:\n"
        for conclusion in self.conclusions:
            summary += f"- {conclusion['conclusion']}\n"
        return summary

class DataAnalysisTool(BaseTool):
    """Ferramenta customizada para análise de dados"""
    
    name: str = "data_analysis_tool"
    description: str = "Ferramenta para realizar análises exploratórias de dados"
    df: Optional[pd.DataFrame] = None
    memory: Optional[object] = None
   
    def _run(self, analysis_type: str) -> str:
        """Executa diferentes tipos de análise"""
        try:
            if analysis_type == "basic_info":
                return self._basic_info()
            elif analysis_type == "correlation_analysis":
                return self._correlation_analysis()
            elif analysis_type == "outlier_detection":
                return self._outlier_detection()
            elif analysis_type == "clustering":
                return self._clustering_analysis()
            else:
                return "Tipo de análise não reconhecido"
        except Exception as e:
            return f"Erro na análise: {str(e)}"
    
    def _basic_info(self) -> str:
        """Informações básicas do dataset"""
        info = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "numeric_columns": list(self.df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(self.df.select_dtypes(include=['object']).columns)
        }
        
        self.memory.dataset_info = info
        self.memory.add_analysis("basic_info", info)
        
        return f"""
        Dataset possui {info['shape'][0]} linhas e {info['shape'][1]} colunas.
        Colunas numéricas: {info['numeric_columns']}
        Colunas categóricas: {info['categorical_columns']}
        Valores ausentes: {info['missing_values']}
        """
    
    def _correlation_analysis(self) -> str:
        """Análise de correlação"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return "Não há colunas numéricas para análise de correlação"
        
        correlation_matrix = numeric_df.corr()
        strong_correlations = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_correlations.append({
                        'var1': correlation_matrix.columns[i],
                        'var2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        self.memory.add_analysis("correlation", strong_correlations)
        
        result = "Correlações fortes encontradas:\n"
        for corr in strong_correlations:
            result += f"- {corr['var1']} vs {corr['var2']}: {corr['correlation']:.3f}\n"
        
        return result if strong_correlations else "Nenhuma correlação forte encontrada (|r| > 0.7)"
    
    def _outlier_detection(self) -> str:
        """Detecção de outliers usando IQR"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        outliers_info = {}
        
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = numeric_df[(numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)]
            outliers_info[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(numeric_df)) * 100,
                'bounds': (lower_bound, upper_bound)
            }
        
        self.memory.add_analysis("outliers", outliers_info)
        
        result = "Outliers detectados:\n"
        for col, info in outliers_info.items():
            if info['count'] > 0:
                result += f"- {col}: {info['count']} outliers ({info['percentage']:.2f}%)\n"
        
        return result
    
    def _clustering_analysis(self) -> str:
        """Análise de clustering"""
        numeric_df = self.df.select_dtypes(include=[np.number]).dropna()
        
        if len(numeric_df.columns) < 2:
            return "Insuficientes colunas numéricas para clustering"
        
        # Normalizar dados
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        cluster_info = {
            'n_clusters': 3,
            'cluster_sizes': pd.Series(clusters).value_counts().to_dict(),
            'inertia': kmeans.inertia_
        }
        
        self.memory.add_analysis("clustering", cluster_info)
        
        return f"""
        Clustering K-means com 3 clusters:
        Tamanhos dos clusters: {cluster_info['cluster_sizes']}
        Inércia: {cluster_info['inertia']:.2f}
        """

class DataAnalysisAgent:
    """Agente especialista em análise de dados"""
    
    def __init__(self, df: pd.DataFrame, db_path: str = "temp_data.db"):
        self.df = df
        self.db_path = db_path
        self.memory = DataAnalysisMemory()
        self.db = None
        self.llm = None
        self.agent_executor = None
        self.chain = None
        self.prompt = None

        self.setup_database()
        self.setup_agent()
    
    def setup_database(self):
        """Configura o banco de dados SQLite"""
        conn = sqlite3.connect(self.db_path)
        self.df.to_sql('data_table', conn, if_exists='replace', index=False)
        conn.close()
        
        self.db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}")
    
    def setup_agent(self):
        """Configura o agente LangChain"""
        # Tente criar o LLM (Azure). Se falhar, llm ficará None.
        try:
            self.llm = AzureChatOpenAI(
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                model="gpt-4o",   # ou "gpt-35-turbo"
                temperature=0,
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            )
            logger.info("AzureChatOpenAI inicializado com sucesso.")
        except Exception as e:
            logger.exception("Não foi possível inicializar AzureChatOpenAI: %s", e)
            self.llm = None

        # Se LLM inicializou, cria toolkit e agent_executor
        if self.llm:
            toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
            try:
                self.agent_executor = create_sql_agent(
                    llm=self.llm,
                    toolkit=toolkit,
                    verbose=True,
                    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    memory=ConversationBufferWindowMemory(k=10)
                )
                logger.info("Agent executor criado com sucesso.")
            except Exception as e:
                logger.exception("Erro ao criar agent_executor: %s", e)
                self.agent_executor = None

            # Prompt e chain também
            self.prompt = ChatPromptTemplate.from_template(
                """
                Você é um especialista em análise de dados.
                Dataset carregado com {rows} linhas e {cols} colunas.
                Colunas disponíveis: {columns}.
                Pergunta do usuário: {question}
                Responda de forma detalhada, trazendo possíveis análises, estatísticas e insights.
                """
            )

            try:
                self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
            except Exception as e:
                logger.exception("Erro ao criar LLMChain: %s", e)
                self.chain = None
        else:
            # Modo fallback (sem LLM) — mantém os atributos definidos
            self.agent_executor = None
            self.chain = None
            self.prompt = ChatPromptTemplate.from_template(
                """
                Você é um especialista em análise de dados (modo offline).
                Dataset carregado com {rows} linhas e {cols} colunas.
                Colunas disponíveis: {columns}.
                Pergunta do usuário: {question}
                Responda de forma detalhada, trazendo possíveis análises, estatísticas e insights.
                """
            )
            logger.warning("LLM não inicializado. Agente pronto em modo offline.")
    
    
    def analyze_data(self, question: str) -> str:
        """Analisa dados baseado na pergunta"""
        if self.agent_executor:
            try:
                response = self.agent_executor.run(
                    f"{question}\nMemória atual: {self.memory.get_memory_summary()}"
                )
                return response
            except Exception as e:
                return f"Erro no agente: {str(e)}"
        else:
            return self._manual_analysis(question)
    
    def _manual_analysis(self, question: str) -> str:
        """Análise manual quando o agente não está disponível"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['básica', 'resumo', 'overview', 'informação']):
            return self._get_basic_summary()
        elif any(word in question_lower for word in ['correlação', 'relação', 'correlation']):
            return self._get_correlation_analysis()
        elif any(word in question_lower for word in ['outlier', 'anomalia', 'atípico']):
            return self._get_outlier_analysis()
        elif any(word in question_lower for word in ['cluster', 'agrupamento', 'padrão']):
            return self._get_clustering_analysis()
        else:
            return "Pergunta não reconhecida. Tente perguntas sobre informações básicas, correlações, outliers ou clustering."
    
    def _get_basic_summary(self) -> str:
        """Retorna resumo básico dos dados"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        summary = f"""
        📊 **RESUMO DO DATASET**
        
        **Dimensões:** {self.df.shape[0]} linhas × {self.df.shape[1]} colunas
        
        **Tipos de Dados:**
        - Numéricas: {list(numeric_cols)} ({len(numeric_cols)} colunas)
        - Categóricas: {list(categorical_cols)} ({len(categorical_cols)} colunas)
        
        **Valores Ausentes:**
        {self.df.isnull().sum().to_string()}
        
        **Estatísticas Descritivas (Numéricas):**
        {self.df[numeric_cols].describe().to_string()}
        """
        
        # Adicionar à memória
        self.memory.add_analysis("basic_summary", {
            "shape": self.df.shape,
            "numeric_cols": list(numeric_cols),
            "categorical_cols": list(categorical_cols)
        })
        
        return summary
    
    def _get_correlation_analysis(self) -> str:
        """Análise de correlação"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return "❌ Não há colunas numéricas para análise de correlação."
        
        corr_matrix = numeric_df.corr()
        
        # Encontrar correlações fortes
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_corr.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_value
                    ))
        
        result = "🔗 **ANÁLISE DE CORRELAÇÃO**\n\n"
        
        if strong_corr:
            result += "**Correlações Fortes (|r| > 0.7):**\n"
            for var1, var2, corr in strong_corr:
                strength = "muito forte" if abs(corr) > 0.9 else "forte"
                direction = "positiva" if corr > 0 else "negativa"
                result += f"- {var1} ↔ {var2}: {corr:.3f} ({strength}, {direction})\n"
        else:
            result += "Nenhuma correlação forte encontrada (|r| > 0.7)\n"
        
        # Adicionar à memória
        self.memory.add_analysis("correlation", strong_corr)
        
        return result
    
    def _get_outlier_analysis(self) -> str:
        """Análise de outliers"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return "❌ Não há colunas numéricas para análise de outliers."
        
        outliers_summary = {}
        
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = numeric_df[(numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)]
            outliers_summary[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(numeric_df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        result = "🚨 **ANÁLISE DE OUTLIERS (Método IQR)**\n\n"
        
        for col, info in outliers_summary.items():
            if info['count'] > 0:
                result += f"**{col}:**\n"
                result += f"  - Outliers detectados: {info['count']} ({info['percentage']:.2f}% dos dados)\n"
                result += f"  - Limites: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]\n\n"
        
        # Adicionar à memória
        self.memory.add_analysis("outliers", outliers_summary)
        
        return result
    
    def _get_clustering_analysis(self) -> str:
        """Análise de clustering"""
        numeric_df = self.df.select_dtypes(include=[np.number]).dropna()
        
        if len(numeric_df.columns) < 2:
            return "❌ Insuficientes colunas numéricas para análise de clustering."
        
        # Padronizar dados
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        cluster_sizes = pd.Series(clusters).value_counts().sort_index()
        
        result = "🔍 **ANÁLISE DE CLUSTERING (K-means, k=3)**\n\n"
        result += f"**Distribuição dos Clusters:**\n"
        for cluster_id, size in cluster_sizes.items():
            percentage = (size / len(clusters)) * 100
            result += f"  - Cluster {cluster_id}: {size} pontos ({percentage:.1f}%)\n"
        
        result += f"\n**Inércia:** {kmeans.inertia_:.2f}"
        result += f"\n**Centros dos Clusters:** {len(kmeans.cluster_centers_)} centros identificados"
        
        # Adicionar à memória
        self.memory.add_analysis("clustering", {
            'cluster_sizes': cluster_sizes.to_dict(),
            'inertia': kmeans.inertia_
        })
        
        return result
    
    def get_conclusions(self) -> str:
        """Gera conclusões baseadas nas análises realizadas"""
        if not self.memory.analyses:
            return "Nenhuma análise foi realizada ainda."
        
        conclusions = []
        
        # Análise do dataset
        if self.memory.dataset_info:
            info = self.memory.dataset_info
            if info.get('missing_values'):
                missing_count = sum([v for v in info['missing_values'].values() if v > 0])
                if missing_count > 0:
                    conclusions.append(f"O dataset possui {missing_count} valores ausentes que podem precisar de tratamento.")
        
        # Conclusões sobre correlações
        for analysis in self.memory.analyses:
            if analysis['type'] == 'correlation' and analysis['result']:
                conclusions.append("Foram identificadas correlações fortes entre variáveis, indicando possíveis relações lineares.")
        
        # Conclusões sobre outliers
        outlier_analyses = [a for a in self.memory.analyses if a['type'] == 'outliers']
        if outlier_analyses:
            latest_outliers = outlier_analyses[-1]['result']
            total_outliers = sum([info['count'] for info in latest_outliers.values()])
            if total_outliers > 0:
                conclusions.append(f"Foram detectados {total_outliers} outliers no dataset que podem afetar as análises.")
        
        # Conclusões sobre clustering
        cluster_analyses = [a for a in self.memory.analyses if a['type'] == 'clustering']
        if cluster_analyses:
            conclusions.append("Os dados mostram padrões de agrupamento, sugerindo diferentes segmentos nos dados.")
        
        # Adicionar conclusões à memória
        final_conclusion = "\n".join([f"• {c}" for c in conclusions])
        if final_conclusion:
            self.memory.add_conclusion(final_conclusion)
        
        return final_conclusion if conclusions else "Com base nas análises realizadas, não foram identificados padrões significativos que requerem atenção especial."

def create_visualizations(df: pd.DataFrame, analysis_type: str):
    """Cria visualizações baseadas no tipo de análise"""
    
    if analysis_type == "distribution":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            fig = go.Figure()
            
            for col in numeric_cols[:4]:  # Máximo 4 colunas
                fig.add_trace(go.Histogram(
                    x=df[col],
                    name=col,
                    opacity=0.7
                ))
            
            fig.update_layout(
                title="Distribuição das Variáveis Numéricas",
                xaxis_title="Valor",
                yaxis_title="Frequência",
                barmode='overlay'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "correlation":
        numeric_df = df.select_dtypes(include=[np.number])
        
        if not numeric_df.empty and len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Matrix de Correlação",
                color_continuous_scale='RdBu_r'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "outliers":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            fig = go.Figure()
            
            for col in numeric_cols[:4]:  # Máximo 4 colunas
                fig.add_trace(go.Box(
                    y=df[col],
                    name=col
                ))
            
            fig.update_layout(
                title="Detecção de Outliers (Box Plots)",
                yaxis_title="Valor"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "scatter":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            
            fig = px.scatter(
                df,
                x=col1,
                y=col2,
                title=f"Gráfico de Dispersão: {col1} vs {col2}"
            )
            
            st.plotly_chart(fig, use_container_width=True)

def main():
    """Função principal do aplicativo"""
    
    # Título e descrição
    st.title("🔍 Especialista em Análise de Dados com IA")
    st.markdown("### Sistema inteligente para análise exploratória de dados com agentes LangChain")
    
    

    # Sidebar para upload e configurações
    with st.sidebar:
        st.header("📁 Upload de Dados")
        
        uploaded_file = st.file_uploader(
            "Faça upload do seu arquivo",
            type=['xls', 'xlsx', 'csv', 'txt'],
            help="Upload de um arquivo para análise"
        )
        
        if uploaded_file is not None:
            try:
                # Carregar dados
                df = pd.read_csv(uploaded_file)
                st.success(f"Arquivo carregado: {uploaded_file.name}")
                st.info(f"Dimensões: {df.shape[0]} linhas × {df.shape[1]} colunas")
                
                # Armazenar no session state
                st.session_state['df'] = df
                st.session_state['uploaded'] = True
                
            except Exception as e:
                st.error(f"Erro ao carregar arquivo: {str(e)}")
                st.session_state['uploaded'] = False
        
        #Caixa de texto para perguntas
        st.header("❓ Diga sua pergunta")
        text_area = st.text_area(
            "Digite sua pergunta sobre os dados após o upload",
            placeholder="Exemplo:\n- Quais são as principais correlações nos dados?",
            height=130,
            help="Seja específico em sua pergunta para obter melhores respostas"
        )
        if st.button("🚀 Enviar Pergunta"):
            if text_area.strip():
                with st.spinner("Processando..."):
                    try:
                        response = st.session_state.agent.analyze_data(text_area)
                        st.success("✅ Resposta do Agente:")
                        st.write(response)
                    except Exception as e:
                        st.error(f"❌ Erro: {str(e)}")
            else:
                st.warning("⚠️ Digite uma pergunta antes de enviar.")    
    
    # Área principal
    if st.session_state.get('uploaded', False) and 'df' in st.session_state:
        df = st.session_state['df']
        
        # Inicializar agente
        if 'agent' not in st.session_state:
            db_path = "temp_data.db"
            st.session_state['agent'] = DataAnalysisAgent(df, db_path)
        
        agent = st.session_state['agent']
        
        # Tabs principais
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Visão Geral", 
            "🤖 Chat com IA", 
            "📈 Visualizações", 
            "💡 Conclusões"
        ])
        
        with tab1:
            st.header("Visão Geral dos Dados")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Informações Básicas")
                st.write(f"**Linhas:** {df.shape[0]}")
                st.write(f"**Colunas:** {df.shape[1]}")
                st.write(f"**Memória:** {df.memory_usage().sum() / 1024**2:.2f} MB")
            
            with col2:
                st.subheader("Tipos de Dados")
                dtype_counts = df.dtypes.value_counts()
                st.write(dtype_counts)
            
            st.subheader("Prévia dos Dados")
            st.dataframe(df.head(10))
            
            st.subheader("Informações Detalhadas")
            st.text(df.info())
            
            st.subheader("Estatísticas Descritivas")
            st.dataframe(df.describe())
        
        with tab2:
            st.header("💬 Chat com o Especialista em Dados")
            
            # Perguntas sugeridas
            st.subheader("Perguntas Sugeridas:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 Resumo básico dos dados"):
                    with st.spinner("Analisando..."):
                        response = agent.analyze_data("Forneça um resumo básico dos dados")
                        st.write(response)
                
                if st.button("🔗 Análise de correlações"):
                    with st.spinner("Calculando correlações..."):
                        response = agent.analyze_data("Analise as correlações entre as variáveis")
                        st.write(response)
            
            with col2:
                if st.button("🚨 Detectar outliers"):
                    with st.spinner("Detectando anomalias..."):
                        response = agent.analyze_data("Detecte outliers nos dados")
                        st.write(response)
                
                if st.button("🔍 Análise de clustering"):
                    with st.spinner("Analisando padrões..."):
                        response = agent.analyze_data("Realize análise de clustering")
                        st.write(response)
            
            st.divider()
            
            # Chat personalizado
            st.subheader("💬 Faça sua pergunta ao especialista:")
            
            # Container para o histórico de chat
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # Exibir histórico de chat
            if st.session_state.chat_history:
                st.subheader("📋 Histórico da Conversa:")
                for i, (q, a) in enumerate(st.session_state.chat_history):
                    with st.expander(f"Pergunta {i+1}: {q[:50]}..."):
                        st.write(f"**🤔 Pergunta:** {q}")
                        st.write(f"**🤖 Resposta:** {a}")
                
                # Botão para limpar histórico
                if st.button("🗑️ Limpar Histórico"):
                    st.session_state.chat_history = []
                    st.rerun()
            
            st.divider()
            
            # Área de entrada de texto
            with st.form("chat_form"):
                question = st.text_area(
                    "Digite sua pergunta sobre os dados:",
                    placeholder="Exemplo:\n- Quais são as principais correlações nos dados?\n- Existem outliers que devo me preocupar?\n- Como estão distribuídas as variáveis?\n- Que padrões você identifica nos dados?",
                    height=100,
                    help="Seja específico em sua pergunta para obter melhores respostas"
                )
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    submit_button = st.form_submit_button("🚀 Enviar Pergunta", use_container_width=True)
                with col2:
                    clear_button = st.form_submit_button("🧹 Limpar", use_container_width=True)
            
            # Processamento da pergunta
            if submit_button and question.strip():
                with st.spinner("🔍 Analisando seus dados e processando a pergunta..."):
                    try:
                        # Adicionar contexto sobre os dados à pergunta
                        enhanced_question = f"""
                        Baseado no dataset carregado com as seguintes características:
                        - Formato: {df.shape[0]} linhas e {df.shape[1]} colunas
                        - Colunas numéricas: {list(df.select_dtypes(include=[np.number]).columns)}
                        - Colunas categóricas: {list(df.select_dtypes(include=['object']).columns)}
                        
                        Pergunta do usuário: {question}
                        
                        Por favor, forneça uma análise detalhada e específica baseada nos dados.
                        """
                        
                        response = agent.analyze_data(enhanced_question)
                        
                        # Adicionar ao histórico
                        st.session_state.chat_history.append((question, response))
                        
                        # Exibir resposta
                        st.success("✅ Análise concluída!")
                        
                        # Container para a resposta mais visível
                        with st.container():
                            st.markdown("### 🤖 Resposta do Especialista:")
                            
                            # Criar tabs para diferentes visualizações da resposta
                            tab_resp, tab_viz = st.tabs(["📝 Resposta", "📊 Visualizações"])
                            
                            with tab_resp:
                                st.markdown(response)
                                
                                # Botões de ação
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    if st.button("👍 Útil"):
                                        st.success("Obrigado pelo feedback!")
                                with col2:
                                    if st.button("🔄 Reformular"):
                                        st.info("Tente fazer a pergunta de forma diferente")
                                with col3:
                                    if st.button("💾 Salvar"):
                                        # Aqui você pode implementar salvamento
                                        st.info("Resposta salva no histórico!")
                            
                            with tab_viz:
                                # Sugerir visualizações baseadas na pergunta
                                question_lower = question.lower()
                                if any(word in question_lower for word in ['correlação', 'relação', 'correlation']):
                                    st.info("💡 Visualização sugerida: Vá para a aba 'Visualizações' e selecione 'Correlações'")
                                elif any(word in question_lower for word in ['outlier', 'anomalia', 'atípico']):
                                    st.info("💡 Visualização sugerida: Vá para a aba 'Visualizações' e selecione 'Outliers'")
                                elif any(word in question_lower for word in ['distribuição', 'histograma']):
                                    st.info("💡 Visualização sugerida: Vá para a aba 'Visualizações' e selecione 'Distribuições'")
                                else:
                                    st.info("💡 Explore a aba 'Visualizações' para gráficos relacionados à sua pergunta")
                    
                    except Exception as e:
                        st.error(f"❌ Erro ao processar a pergunta: {str(e)}")
                        st.info("💡 Tente reformular sua pergunta ou use uma das perguntas sugeridas")
            
            elif submit_button and not question.strip():
                st.warning("⚠️ Por favor, digite uma pergunta antes de enviar.")
            
            elif clear_button:
                st.info("✅ Campo de pergunta limpo!")
            
            # Seção de dicas
            with st.expander("💡 Dicas para fazer boas perguntas"):
                st.markdown("""
                **🎯 Para obter melhores respostas:**
                
                **✅ Perguntas específicas funcionam melhor:**
                - "Qual a correlação entre idade e salário?"
                - "Quantos outliers existem na coluna de vendas?"
                - "Quais variáveis têm maior impacto no resultado?"
                
                **❌ Evite perguntas muito genéricas:**
                - "Me fale sobre os dados"
                - "O que você acha?"
                - "Análise tudo"
                
                **🔍 Tipos de análise disponíveis:**
                - **Descritiva:** estatísticas, distribuições, resumos
                - **Correlação:** relacionamentos entre variáveis
                - **Outliers:** valores atípicos e anomalias
                - **Clustering:** padrões e agrupamentos
                - **Tendências:** análise temporal e padrões
                
                **💬 Exemplos de perguntas por categoria:**
                
                **📊 Análise Descritiva:**
                - "Quais são as estatísticas básicas de cada variável?"
                - "Como estão distribuídos os dados?"
                - "Há muitos valores ausentes?"
                
                **🔗 Análise de Relacionamentos:**
                - "Existe correlação entre X e Y?"
                - "Quais variáveis se relacionam mais fortemente?"
                - "Como X influencia Y?"
                
                **🚨 Detecção de Anomalias:**
                - "Há outliers nos dados?"
                - "Quais valores são considerados atípicos?"
                - "Os outliers afetam a análise?"
                
                **🔍 Identificação de Padrões:**
                - "Existem grupos naturais nos dados?"
                - "Há padrões temporais?"
                - "Que segmentos posso identificar?"
                """)
            
            # Atalhos para perguntas comuns
            st.subheader("⚡ Análises Rápidas:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📈 Análise Completa", help="Análise exploratória completa dos dados"):
                    quick_question = "Faça uma análise exploratória completa dos dados, incluindo estatísticas descritivas, correlações, outliers e padrões identificados."
                    with st.spinner("Realizando análise completa..."):
                        response = agent.analyze_data(quick_question)
                        st.session_state.chat_history.append((quick_question, response))
                        st.markdown("### 🤖 Análise Completa:")
                        st.markdown(response)
            
            with col2:
                if st.button("🔗 Correlações", help="Análise de correlações entre variáveis"):
                    quick_question = "Analise as correlações entre todas as variáveis numéricas e identifique os relacionamentos mais significativos."
                    with st.spinner("Analisando correlações..."):
                        response = agent.analyze_data(quick_question)
                        st.session_state.chat_history.append((quick_question, response))
                        st.markdown("### 🔗 Análise de Correlações:")
                        st.markdown(response)
            
            with col3:
                if st.button("⚠️ Problemas", help="Identifica problemas nos dados"):
                    quick_question = "Identifique possíveis problemas nos dados como outliers, valores ausentes, inconsistências e outros issues que precisam de atenção."
                    with st.spinner("Identificando problemas..."):
                        response = agent.analyze_data(quick_question)
                        st.session_state.chat_history.append((quick_question, response))
                        st.markdown("### ⚠️ Problemas Identificados:")
                        st.markdown(response)
        
        with tab3:
            st.header("📈 Visualizações")
            
            viz_type = st.selectbox(
                "Selecione o tipo de visualização:",
                ["Distribuições", "Correlações", "Outliers", "Dispersão"]
            )
            
            if st.button("Gerar Visualização"):
                with st.spinner("Criando visualização..."):
                    if viz_type == "Distribuições":
                        create_visualizations(df, "distribution")
                    elif viz_type == "Correlações":
                        create_visualizations(df, "correlation")
                    elif viz_type == "Outliers":
                        create_visualizations(df, "outliers")
                    elif viz_type == "Dispersão":
                        create_visualizations(df, "scatter")
        
        with tab4:
            st.header("💡 Conclusões e Insights")
            
            if st.button("🧠 Gerar Conclusões"):
                with st.spinner("Analisando todos os dados e gerando insights..."):
                    conclusions = agent.get_conclusions()
                    
                    if conclusions:
                        st.success("Conclusões geradas com base nas análises realizadas:")
                        st.write(conclusions)
                    else:
                        st.info("Realize algumas análises primeiro para gerar conclusões.")
            
            # Mostrar memória do agente
            st.subheader("📚 Memória do Agente")
            if agent.memory.analyses:
                st.write(f"**Análises realizadas:** {len(agent.memory.analyses)}")
                st.write(f"**Conclusões armazenadas:** {len(agent.memory.conclusions)}")
                
                with st.expander("Ver detalhes da memória"):
                    st.text(agent.memory.get_memory_summary())
            else:
                st.info("Nenhuma análise realizada ainda. Use o chat para fazer perguntas sobre os dados.")
    
    else:
        # Página inicial quando não há dados carregados
        st.info("👈 Faça upload de um arquivo CSV na sidebar para começar a análise")
        
        # Tutorial de uso
        st.subheader("🚀 Como usar este sistema:")
        
        st.markdown("""
        1. **Upload de Dados**: Carregue seu arquivo CSV usando a sidebar
        2. **Visão Geral**: Explore informações básicas sobre seus dados
        3. **Chat com IA**: Faça perguntas em linguagem natural sobre os dados
        4. **Visualizações**: Gere gráficos automáticos para diferentes tipos de análise
        5. **Conclusões**: Obtenha insights inteligentes baseados em todas as análises
        
        ### 🤖 Capacidades do Agente:
        - Análise exploratória completa (EDA)
        - Detecção de padrões e tendências
        - Identificação de outliers e anomalias
        - Análise de correlações entre variáveis
        - Clustering e segmentação
        - Conclusões inteligentes com memória persistente
        
        ### 📊 Tipos de Perguntas que pode fazer:
        - "Quais são os principais insights destes dados?"
        - "Existem correlações fortes entre as variáveis?"
        - "Como estão distribuídos os dados?"
        - "Há outliers que devo me preocupar?"
        - "Quais padrões você encontrou?"
        """)
        
        # Exemplo de dataset
        st.subheader("📋 Dataset de Exemplo")
        st.markdown("Experimente com este dataset de exemplo:")
        
        if st.button("Gerar Dataset de Exemplo"):
            # Criar dataset de exemplo
            np.random.seed(42)
            n_samples = 1000
            
            example_data = {
                'idade': np.random.randint(18, 80, n_samples),
                'salario': np.random.exponential(50000, n_samples) + 30000,
                'experiencia': np.random.randint(0, 40, n_samples),
                'satisfacao': np.random.uniform(1, 11, n_samples),
                'departamento': np.random.choice(['TI', 'Vendas', 'Marketing', 'RH', 'Financeiro'], n_samples),
                'performance': np.random.normal(7, 2, n_samples)
            }
            
            # Adicionar algumas correlações
            example_data['salario'] += example_data['experiencia'] * 1000
            example_data['satisfacao'] += (example_data['performance'] - 7) * 0.5
            
            example_df = pd.DataFrame(example_data)
            
            # Salvar arquivo temporário
            example_df.to_csv('exemplo_dataset.csv', index=False)
            
            st.success("Dataset de exemplo criado!")
            st.dataframe(example_df.head())
            
            # Carregar automaticamente
            st.session_state['df'] = example_df
            st.session_state['uploaded'] = True
            st.rerun()

if __name__ == "__main__":
    # Inicializar session state
    if 'uploaded' not in st.session_state:
        st.session_state['uploaded'] = False
    
    main()