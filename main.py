import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from langchain.agents import Agent, Task
from langchain.graph import Graph
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Gerando dataset simulado
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=180, freq='D')
produto_ids = [f"produto_{i}" for i in range(1, 6)]
setores = ['Setor 01', 'Setor 02', 'Setor 03']
prateleiras = ['Prateleira X', 'Prateleira Y', 'Prateleira Z']
data = []

for produto in produto_ids:
    vendas = np.random.poisson(lam=20, size=len(dates))
    estoque = np.random.poisson(lam=50, size=len(dates))
    validade = pd.date_range(start='2023-06-01', periods=len(dates), freq='D') + pd.to_timedelta(np.random.randint(1, 180), unit='D')
    localizacao = [(np.random.choice(setores), np.random.choice(prateleiras)) for _ in range(len(dates))]
    data.extend(zip(dates, [produto]*len(dates), vendas, estoque, validade, localizacao))

df = pd.DataFrame(data, columns=['data', 'produto_id', 'vendas', 'estoque', 'validade', 'localizacao'])

df[['setor', 'prateleira']] = pd.DataFrame(df['localizacao'].tolist(), index=df.index)
df = df.drop(columns=['localizacao'])

df.to_csv('vendas_simuladas_expandido.csv', index=False)

# Visualizando o dataset
df.head()


##############
df = pd.read_csv('vendas_simuladas_expandido.csv')
df['data'] = pd.to_datetime(df['data'])
df['validade'] = pd.to_datetime(df['validade'])

# Função para preprocessar dados
def preprocess_data(df, produto_id):
    df_produto = df[df['produto_id'] == produto_id]
    df_produto = df_produto.set_index('data').asfreq('D').fillna(method='ffill')
    return df_produto

produto_id = 'produto_1'
df_produto = preprocess_data(df, produto_id)

df_produto.head()

###################

df = pd.read_csv('vendas_simuladas_expandido.csv')
df['data'] = pd.to_datetime(df['data'])
df['validade'] = pd.to_datetime(df['validade'])

# Função para preprocessar dados
def preprocess_data(df, produto_id):
    df_produto = df[df['produto_id'] == produto_id]
    df_produto = df_produto.set_index('data').asfreq('D').fillna(method='ffill')
    return df_produto

produto_id = 'produto_1'
df_produto = preprocess_data(df, produto_id)

df_produto.head()

############

# Feature engineering
df_produto['dia_da_semana'] = df_produto.index.dayofweek
df_produto['dia_do_mes'] = df_produto.index.day
df_produto['mes'] = df_produto.index.month

X = df_produto[['estoque', 'dia_da_semana', 'dia_do_mes', 'mes']]
y = df_produto['vendas']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Treinando o modelo
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Avaliando o modelo
y_pred = modelo.predict(X_test)

plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Vendas Reais')
plt.plot(y_test.index, y_pred, label='Previsões de Vendas')
plt.legend()
plt.show()

#############
# Previsão para os próximos 30 dias
futuro = pd.date_range(start=df_produto.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
futuro_df = pd.DataFrame(futuro, columns=['data'])
futuro_df['estoque'] = np.random.poisson(lam=50, size=len(futuro))
futuro_df['dia_da_semana'] = futuro_df['data'].dt.dayofweek
futuro_df['dia_do_mes'] = futuro_df['data'].dt.day
futuro_df['mes'] = futuro_df['data'].dt.month

X_futuro = futuro_df[['estoque', 'dia_da_semana', 'dia_do_mes', 'mes']]
futuro_df['vendas_previstas'] = modelo.predict(X_futuro)

# Notificação aos fornecedores
futuro_df['ajuste_estoque'] = futuro_df['vendas_previstas'] * 1.2  # Ajuste de 20% acima da previsão

# Exibir decisões
futuro_df[['data', 'vendas_previstas', 'ajuste_estoque']].head()

#########

# Produtos prestes a vencer
hoje = pd.Timestamp.today()
validade_alerta = hoje + pd.Timedelta(days=30)  # Alerta para produtos que vencem nos próximos 30 dias

df_validade = df_produto[df_produto['validade'] <= validade_alerta]
df_validade = df_validade[['validade', 'estoque', 'setor', 'prateleira']]

# Exibir produtos prestes a vencer
df_validade.head()

###########

# Definindo a tarefa de previsão de demanda
class PrevisaoDemanda(Task):
    def run(self, dados):
        produto_id = dados.get('produto_id', 'produto_1')
        df_produto = preprocess_data(df, produto_id)
        futuro_df['vendas_previstas'] = modelo.predict(X_futuro)
        futuro_df['ajuste_estoque'] = futuro_df['vendas_previstas'] * 1.2
        return futuro_df[['data', 'vendas_previstas', 'ajuste_estoque']].to_dict()

# Definindo a tarefa de notificação a fornecedores
class NotificacaoFornecedores(Task):
    def run(self, dados):
        produto_id = dados.get('produto_id', 'produto_1')
        futuro_df = dados.get('futuro_df')
        produtos_faltando = futuro_df[futuro_df['ajuste_estoque'] > futuro_df['estoque']]
        return produtos_faltando[['data', 'ajuste_estoque']].to_dict()

# Definindo a tarefa de gestão de estoque
class GestaoEstoque(Task):
    def run(self, dados):
        produto_id = dados.get('produto_id', 'produto_1')
        df_produto = preprocess_data(df, produto_id)
        hoje = pd.Timestamp.today()
        validade_alerta = hoje + pd.Timedelta(days=30)
        df_validade = df_produto[df_produto['validade'] <= validade_alerta]
        return df_validade[['validade', 'estoque', 'setor', 'prateleira']].to_dict()

# Criando os agentes
agente_previsao = Agent(name="Agente de Previsão de Demanda")
agente_previsao.add_task(PrevisaoDemanda())

agente_fornecedores = Agent(name="Agente de Notificação a Fornecedores")
agente_fornecedores.add_task(NotificacaoFornecedores())

agente_estoque = Agent(name="Agente de Gestão de Estoque")
agente_estoque.add_task(GestaoEstoque())

# Definindo o fluxo no LangGraph
grafo = Graph(name="Fluxo de Previsão e Gestão de Estoque")
grafo.add_node(agente_previsao)
grafo.add_node(agente_fornecedores)
grafo.add_node(agente_estoque)

# Executando a previsão e notificações
resultado_previsao = agente_previsao.run({'produto_id': 'produto_1'})
resultado_notificacao = agente_fornecedores.run({'produto_id': 'produto_1', 'futuro_df': resultado_previsao})
resultado_estoque = agente_estoque.run({'produto_id': 'produto_1'})

print("Previsão de Demanda:", resultado_previsao)
print("Notificação a Fornecedores:", resultado_notificacao)
print("Gestão de Estoque:", resultado_estoque)


