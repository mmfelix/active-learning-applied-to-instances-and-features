import pandas as pd
from scipy.io import arff
import os
import re


def check_duplicate_fields(filename: str, output_path: str) -> None:
    """Checks and renames duplicate fields in an ARFF file.

    Parameters
    ----------
    filename : str
        _description_
    output_path : str
        _description_
    """

    with open(filename, "r") as f:
        lines = f.readlines()

    seen = {}
    new_lines = []
    for line in lines:
        if line.lower().startswith('@attribute'):
            parts = re.split(r'\s+', line.strip(), maxsplit=2)
            if len(parts) < 3:
                new_lines.append(line)
                continue
            attr_name = parts[1]

            # Renombrar si ya apareció
            if attr_name in seen:
                seen[attr_name] += 1
                new_name = f"{attr_name}_{seen[attr_name]}"
            else:
                seen[attr_name] = 1
                new_name = attr_name

            new_line = f"@attribute {new_name} {parts[2]}\n"
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    # Guardar archivo corregido
    output_file = os.path.join(output_path, os.path.basename(filename))
    with open(output_file, "w") as f:
        f.writelines(new_lines)

    print(f"Archivo corregido guardado como: {output_file}")

def read_arff(filename: str) -> pd.DataFrame:
    """Reads an ARFF file and converts it to a Pandas DataFrame.

    Parameters
    ----------
    filename : str
        The name of the ARFF file to be read.

    Returns
    -------
    pd.DataFrame
        The DataFrame created from the ARFF file.
    """
    data, meta = arff.loadarff(filename)
    df = pd.DataFrame(data)

    for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode('utf-8')

    return df

def class_convertion(df: pd.DataFrame) -> pd.DataFrame:
    """Converts class labels in the DataFrame to integers.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing class labels to be converted.

    Returns
    -------
    pd.DataFrame
        The DataFrame with class labels converted to integers.
    """
    df[df.columns[-1]] = df[df.columns[-1]].astype('category')
    df[df.columns[-1]] = df[df.columns[-1]].cat.codes

    return df  
    

if __name__ == "__main__":
    original_path = './datasets/original/'
    preprocessed_path = './datasets/pre-processed/'
    processed_path = './datasets/processed/'
    
    for filename in os.listdir(original_path):
        if filename.endswith('.arff'):
            print(filename)
            check_duplicate_fields(os.path.join(original_path, filename), preprocessed_path)         

    for filename in os.listdir(preprocessed_path):
        if filename.endswith('.arff'):
            print(filename)
            df = read_arff(os.path.join(preprocessed_path, filename))
            base_name = os.path.splitext(os.path.basename(filename))[0]
            output_name = os.path.join(processed_path, base_name)
            df.to_parquet(f'{output_name}.parquet', engine='pyarrow')