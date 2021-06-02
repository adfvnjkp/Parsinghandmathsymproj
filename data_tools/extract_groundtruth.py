import argparse
import csv
import glob
import os
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

doc_namespace = "{http://www.w3.org/2003/InkML}"

def extract_truth2(file_path):
    '''
    extract symbol truth
    '''
    s = ""
    try:
        
        inkml_input = BeautifulSoup(open(file_path), "xml")


        trace_groups = {}
        for st in inkml_input.find_all('traceGroup'):
            label = st.find(name='annotation', attrs={'type': 'truth'}).text

            for id in st.find_all(name='traceView'):
                trace_groups[int(id['traceDataRef'])] = label
        
        sorted_keys = sorted(trace_groups.keys())

   
        
        for i in sorted_keys:
            s += str(trace_groups[i]) + " "
    
    except:
        print('unable to handle')
    
    return s

def extract_truth3(file_path):
    '''
    only for test datset
    '''
    s = ""
    try:
        
        inkml_input = BeautifulSoup(open(file_path), "xml")

        trace = {}
        for st in inkml_input.find_all('trace'):
            s +="1 "
    
    except:
        print('unable to handle')
    
    return s

def extract_truth(file_path):
    '''
    extract formula
    '''
    root = ET.parse(file_path).getroot()
    annotations = root.findall(doc_namespace + "annotation")
    truths = [ann for ann in annotations if ann.get("type") == "truth"]
    if len(truths) != 1:
        raise Exception(
            "{} does not contain a ground truth annotation".format(file_path)
        )
    return truths[0].text


def create_tsv(path, output="groundtruth.tsv"):
    files = glob.glob(os.path.join(path, "*.inkml"))
    with open(output, "w", newline="") as fd:
        writer = csv.writer(fd, delimiter="\t")
        for f in files:
            rel_path = os.path.relpath(f, path)
            reference_name = os.path.splitext(rel_path)[0]
            truth = extract_truth3(f)
            # Remove $ because the entire forumla is surrounded by it.
            #truth = truth.replace("$", "")
            writer.writerow([reference_name, truth])


if __name__ == "__main__":
    """
    extract_groundtruth path/to/dataset [-o OUTPUT]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        default="groundtruth.tsv",
        help="Output path of the TSV file",
    )
    parser.add_argument(
        "directory", nargs=1, help="Directory to data with ground truth"
    )
    args = parser.parse_args()
    create_tsv(args.directory[0], args.output)
