from transformers import AutoModel

def model_loader(models: list) -> list:
    downloaded_models = []
    for model_path in models:
        print('Downloading model', model_path, '...')
        try:
            model = AutoModel.from_pretrained(model_path)
            downloaded_models.append(model)
        except OSError:
            print(model_path, 'not found.')

    return downloaded_models

def save_models(models: list, model_names: list) -> None:
    for index, model in enumerate(models):
        model.save_pretrained('models/' + model_names[index])
    return


if __name__ == '__main__': 
    with open('models.txt', 'r') as file:
        models = file.read().splitlines()
    file.close()

    downloaded_models = model_loader(models)
    save_models(downloaded_models, models)