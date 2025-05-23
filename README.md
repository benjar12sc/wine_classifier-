# Wine Classifier Project

## Notebooks
- `notebooks/wine_exploration.ipynb`: Explore and visualize the wine dataset.
- `notebooks/wine_random_forest.ipynb`: Train and evaluate a Random Forest classifier, including feature importance and overfitting mitigation.
- `notebooks/wine_svm.ipynb`: Train and evaluate a Support Vector Machine (SVM) classifier, including hyperparameter tuning and feature importance.

## Training the Model

You can train the SVM model using the provided script:

```sh
python scripts/train_svm.py
```

Or with Docker Compose:

```sh
docker-compose run --rm train
```

This will save the trained model and scaler to the `models/` directory.

## Serving the API

You can serve the trained model using FastAPI:

```sh
python scripts/serve_svm_api.py
```

Or with Docker Compose:

```sh
docker-compose up api
```

The API will be available at `http://localhost:8000`.

## Example Request

Send a POST request to `/predict` with the following JSON body:

```json
{
  "proline": 1000,
  "od280_od315_of_diluted_wines": 3.0,
  "color_intensity": 5.0,
  "flavanoids": 2.5,
  "alcohol": 13.0
}
```

## Example Response

```json
{
  "prediction": 1,
  "probabilities": [0.01, 0.97, 0.02],
  "class_names": [0, 1, 2]
}
```

- `prediction`: The predicted class label.
- `probabilities`: The probability for each class.
- `class_names`: The class labels corresponding to the probabilities.

## Authentication & Admin

- The `/predict` endpoint requires a Bearer token in the `Authorization` header. Tokens are managed in a SQLite database.
- Admins can add new tokens via the `/admin/add_token` endpoint using the admin token (see code/config).
- Example admin request:

```sh
curl -X POST "http://localhost:8000/admin/add_token" \
  -H "Authorization: Bearer adminsupersecret" \
  -H "Content-Type: application/json" \
  -d '{"new_token": "myusertoken", "rate_limit_seconds": 60}'
```

## License

This project is licensed under an MIT-style license with attribution required. See LICENSE for details.
