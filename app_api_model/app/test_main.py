from fastapi.testclient import TestClient

from .main import app

client = TestClient(app)


def test_get_threshold():
    response = client.post("/get_threshold")
    assert response.status_code == 200

    # On vérifie que la fonction retourne bien un nombre de type float
    assert isinstance(response.json()['data'], float)
    
    # On vérifie que le float retourné est bien entre 0 et 1
    data_value = response.json()['data']
    assert 0 <= data_value <= 1

def test_shap_model():
    response = client.post("/shap_model")
    assert response.status_code == 200

    # On vérifie que la fonction retourne bien un nombre de type float
    assert isinstance(response.json()['data'], dict)

def test_risque_th_model():
    response = client.post("/risque_th_model")
    assert response.status_code == 200

    # On vérifie que la fonction retourne bien un nombre de type float
    assert isinstance(response.json()['data'], dict)
    
def test_client_th_model():
    response = client.post("/client_th_model")
    assert response.status_code == 200

    # On vérifie que la fonction retourne bien un nombre de type float
    assert isinstance(response.json()['data'], dict)

def test_importance_client():
    
    input_data = {"key1": [0, 0], "key2": [0, 0]}

    response = client.post("/importance_client", json={"data":input_data})
    assert response.status_code == 200

    # On vérifie que "data" est bien dans la réponse
    assert 'data' in response.json()

    # On vérifie que la fonction retourne bien un nombre de type float
    assert isinstance(response.json()['data'], dict)
    
def test_prediction():
    
    input_data = {"key1": [0, 0], "key2": [0, 0]}

    response = client.post("/prediction", json={"data":input_data, "thresh":0.5})
    assert response.status_code == 200

    # On vérifie que "data" est bien dans la réponse
    assert 'data' in response.json()

    # On vérifie que la fonction retourne bien un nombre de type float
    assert isinstance(response.json()['data'], dict)

def test_scoring():
    
    input_data = {"key1": [0, 0], "key2": [0, 0]}

    response = client.post("/scoring", json={"data":input_data, "thresh":0.5})
    assert response.status_code == 200

    # On vérifie que "data" est bien dans la réponse
    assert 'data' in response.json()

    # On vérifie que la fonction retourne bien un nombre de type float
    assert isinstance(response.json()['data'], dict)