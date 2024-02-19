from langchain_community.document_loaders.csv_loader import CSVLoader

def load_train_data():
    loader = CSVLoader(file_path='/home/minahwang2001/data/train_data.csv',encoding='utf-8')
    data = loader.load()
    return data