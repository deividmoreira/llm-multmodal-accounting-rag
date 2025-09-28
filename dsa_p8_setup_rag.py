# Projeto 8 - IA Generativa Multimodal com Agentic RAG e LangGraph Para Análise Contábil
# Módulo de RAG

# Imports
import os
import faiss
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings

# Define o diretório onde estão os manuais de contabilidade em PDF
PDF_SOURCE_DIR = "dsa_pdfs_contabilidade"

# Define o caminho onde o índice vetorial FAISS será salvo
VECTORSTORE_PATH = "dsa_faiss_index_contabilidade"

# Se o diretório de PDFs não existir, cria-o e encerra o script
if not os.path.exists(PDF_SOURCE_DIR):
    
    # Cria o diretório para armazenar os PDFs
    os.makedirs(PDF_SOURCE_DIR)
    
    # Informa ao usuário que o diretório foi criado
    print(f"Diretório '{PDF_SOURCE_DIR}' criado.")
    
    # Orienta o usuário a adicionar os manuais em PDF nesse diretório
    print("Por favor, adicione seus manuais de contabilidade em PDF nele.")
    
    # Encerra a execução do script
    exit()

# Função principal que carrega PDFs, divide em chunks e gera o vector store
def dsa_cria_vectordb():

    # Indica que o carregamento dos PDFs está começando
    print(f"Carregando PDFs do diretório: {PDF_SOURCE_DIR}")
    
    # Verifica se há arquivos no diretório de PDFs
    if not os.listdir(PDF_SOURCE_DIR):
         
         # Emite um aviso caso o diretório esteja vazio
         print(f"AVISO: O diretório '{PDF_SOURCE_DIR}' está vazio.")
         pass 

    try:
        
        # Inicializa o loader que percorre o diretório recursivamente
        pdf_loader = PyPDFDirectoryLoader(PDF_SOURCE_DIR, recursive = True)
        
        # Carrega todos os documentos PDF encontrados
        documents = pdf_loader.load()
        
        # Se não houve documentos carregados, avisa o usuário
        if not documents:
            print(f"Nenhum documento PDF encontrado ou carregado de '{PDF_SOURCE_DIR}'. Verifique os arquivos.")
        else:
            # Informa quantas páginas/documentos foram carregados
            print(f"Carregados {len(documents)} páginas/documentos PDF.")

    except Exception as e:
        
        # Mostra o erro caso o carregamento falhe
        print(f"Erro ao carregar PDFs: {e}")
        
        # Retorna False para sinalizar falha crítica
        return False

    # Indica o início da divisão dos documentos em pedaços menores (chunks)
    print("Dividindo documentos em chunks...")
    
    # Configura o splitter de texto baseado em caracteres
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,      # número máximo de caracteres por chunk
        chunk_overlap = 150     # sobreposição de caracteres entre chunks
    )

    # Executa a divisão dos documentos
    docs_split = text_splitter.split_documents(documents)
    
    # Se nenhum chunk foi gerado, avisa o usuário
    if not docs_split:
        print("Nenhum texto foi extraído dos PDFs para dividir.")
    else:
        # Informa a quantidade de chunks gerados
        print(f"Documentos divididos em {len(docs_split)} chunks.")

    # Indica que o modelo de embedding será inicializado
    print("Inicializando modelo de embedding (FastEmbed)...")
    
    # Cria a instância do modelo de embedding FastEmbed
    embedding_model = FastEmbedEmbeddings(model_name = "BAAI/bge-small-en-v1.5")

    # Indica o início da criação do índice FAISS
    print("Criando índice vetorial FAISS...")
    try:

        # Se não houver chunks, cria um índice vazio
        if not docs_split:
             
             print("Criando um índice FAISS vazio pois não há documentos.")
             # Gera embeddings dummy para obter a dimensão correta
             dummy_embeddings = embedding_model.embed_documents(["dummy text"])
             dimension = len(dummy_embeddings[0])
             
             # Cria um índice FAISS baseado em distância L2
             index = faiss.IndexFlatL2(dimension)
             
             # Inicializa o vector store vazio
             vector_store = FAISS(embedding_function = embedding_model.embed_query,
                                  index = index,
                                  docstore = {},
                                  index_to_docstore_id = {})

        else:
            
            # Cria o vector store a partir dos chunks gerados
            vector_store = FAISS.from_documents(docs_split, embedding_model)
            
            # Confirma que o índice foi criado em memória
            print("Índice FAISS criado na memória com documentos.")

        # Salva localmente o índice FAISS no caminho especificado
        vector_store.save_local(VECTORSTORE_PATH)
        
        # Informa ao usuário que o índice foi salvo
        print(f"Índice FAISS salvo localmente em: {VECTORSTORE_PATH}")
        
        # Retorna True indicando sucesso
        return True

    except Exception as e:
        # Mostra o erro caso a criação ou salvamento do índice falhe
        print(f"Erro ao criar ou salvar o índice FAISS: {e}")
        # Retorna False para sinalizar falha
        return False

# Bloco de execução principal do script
if __name__ == "__main__":
    
    # Informa que o processo de configuração do RAG está iniciando
    print("\nIniciando processo de configuração do RAG de Contabilidade...")
    
    # Chama a função de criação do vector store e verifica o resultado
    if dsa_cria_vectordb():
        # Mensagens exibidas em caso de sucesso
        print("Configuração do RAG de Contabilidade concluída com sucesso!")
        print(f"O índice vetorial está salvo em '{VECTORSTORE_PATH}'.")
        print(f"Certifique-se de ter seus PDFs na pasta '{PDF_SOURCE_DIR}'.\n")
    else:
        # Mensagem exibida em caso de falha no processo
        print("\nA configuração do RAG de Contabilidade falhou. Verifique os erros acima.")


