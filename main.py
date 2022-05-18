from timeit import default_timer as timer
from datetime import timedelta
import colorama
from colorama import Style, Fore
import pandas
import scipy.stats
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
import mlsp


def main():
    start_time = timer()

    # Read CSV
    data = pandas.read_csv("./data/diamonds.csv")

    # First look
    mlsp.misc.print_title("First look")
    mlsp.df.first_look(data)

    # Missing values
    mlsp.misc.print_title("Missing values")
    mlsp.df.missing_values(data)

    # Merge [x, y, z] into `volume`
    mlsp.misc.print_title("Merge [x, y, z] into `volume`")
    data["volume"] = data["x"] * data["y"] * data["z"]
    data.drop(columns=["x", "y", "z"], inplace=True)
    print(data.sample(n=5))

    # Study
    mlsp.misc.print_title("Study")
    features = ["carat", "depth", "table", "volume", "price"]

    analyze = data.describe().T
    analyze["dispersion"] = data[features].std() / data[features].mean() * 100
    analyze["standard error"] = data[features].sem()
    analyze["skewness"] = data[features].skew()
    analyze["kurtosis"] = data[features].kurtosis()

    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    is_outliers = ((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr)))
    analyze["outliers"] = is_outliers.sum(axis=0) / data.shape[0] * 100  # Percent of outliers per column

    pandas.set_option("display.max_columns", None)
    print(f"Analyze:\n{Style.DIM}{Fore.WHITE}{analyze}")
    pandas.set_option("display.max_columns", 5)

    # Splitting dataset
    mlsp.misc.print_title("Splitting dataset")
    x_train, x_test, y_train, y_test = mlsp.df.split_train_test(data, y_label="price", test_size=0.20)
    print(f"Train data: {Fore.LIGHTGREEN_EX}{x_train.shape}")
    print(f"Test data: {Fore.LIGHTGREEN_EX}{x_test.shape}")

    # Transform numeric and categorical values
    mlsp.misc.print_title("Transform numeric and categorical values")
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant")),
        ("encoder", OrdinalEncoder())
    ])

    numeric_features = ["carat", "depth", "table", "volume"]
    categorical_features = ["cut", "color", "clarity"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features)
        ]
    )

    # Get model (Linear Regression)
    mlsp.misc.print_title("Get model (Linear Regression)")
    mlsp.models.linear_model.linear_regression_model(
        preprocessor,
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test
    )

    # Get model (KNeighbors Regression)
    mlsp.misc.print_title("Get model (KNeighbors Regression)")
    mlsp.models.neighbors.k_neighbors_regressor(
        preprocessor,
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test
    )

    # Program end
    end_time = timer()
    elapsed_time = timedelta(seconds=end_time - start_time)
    print(f"\n{Fore.GREEN}Successful processing of diamonds dataset in {elapsed_time}.")


if __name__ == "__main__":
    colorama.init(autoreset=True)
    main()
