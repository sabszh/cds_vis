import os
import argparse
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        Namespace: An object containing parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Detect faces in historical newspapers.")
    parser.add_argument("data_path", type=str, help="Path to the folder containing newspaper images.")
    return parser.parse_args()


def detect_faces(data_path):
    """
    Detect faces in historical newspapers.

    Args:
        data_path (str): Path to the folder containing newspaper images.

    Returns:
        defaultdict: A nested dictionary containing the total number of faces per newspaper per decade.
    """
    mtcnn = MTCNN(keep_all=True)
    results = defaultdict(lambda: defaultdict(int))  # {newspaper: {decade: total_faces}}

    for newspaper_folder in os.listdir(data_path):
        newspaper_path = os.path.join(data_path, newspaper_folder)
        if os.path.isdir(newspaper_path):
            newspaper_name = newspaper_folder.upper()

            for filename in os.listdir(newspaper_path):
                if filename.endswith(".jpg"):
                    year = int(filename.split("-")[1])
                    decade = year // 10 * 10
                    img_path = os.path.join(newspaper_path, filename)
                    img = Image.open(img_path)
                    boxes, _ = mtcnn.detect(img)

                    if boxes is not None:
                        num_faces = len(boxes)
                        results[newspaper_name][decade] += num_faces

    return results


def group_by_decade(results):
    """
    Group results by decade.

    Args:
        results (defaultdict): A nested dictionary containing the total number of faces per newspaper per decade.

    Returns:
        defaultdict: A nested dictionary containing the total number of faces per newspaper per decade, with zero values removed.
    """
    grouped_results = defaultdict(dict)  # {newspaper: {decade: total_faces}}
    for newspaper, data in results.items():
        for decade, total_faces in data.items():
            if total_faces > 0:
                grouped_results[newspaper][decade] = total_faces

    return grouped_results


def save_to_csv(grouped_results):
    """
    Save results to CSV files.

    Args:
        grouped_results (defaultdict): A nested dictionary containing the total number of faces per newspaper per decade.
    """
    for newspaper, data in grouped_results.items():
        df = pd.DataFrame.from_dict(data, orient='index', columns=['Total Faces'])
        df.index.name = 'Decade'
        df.to_csv(f"{newspaper}_faces_per_decade.csv")


def plot_results(grouped_results):
    """
    Plot the percentage of pages with faces per decade.

    Args:
        grouped_results (defaultdict): A nested dictionary containing the total number of faces per newspaper per decade.
    """
    for newspaper, data in grouped_results.items():
        decades = list(data.keys())
        total_faces = list(data.values())
        total_pages = [100] * len(decades)
        percentages = [faces / total * 100 for faces, total in zip(total_faces, total_pages)]

        plt.plot(decades, percentages, label=newspaper)

    plt.xlabel("Decade")
    plt.ylabel("Percentage of Pages with Faces")
    plt.title("Percentage of Pages with Faces per Decade")
    plt.legend()
    plt.grid(True)
    plt.savefig("faces_per_decade_plot.png")
    plt.show()


def main():
    """
    Main function.
    """
    args = parse_arguments()
    data_path = args.data_path

    results = detect_faces(data_path)
    grouped_results = group_by_decade(results)
    save_to_csv(grouped_results)
    plot_results(grouped_results)


if __name__ == "__main__":
    main()
