"""Download and extract a bounded real paired Facades subset from Hugging Face."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

SOURCE = 'https://huggingface.co/datasets/huggan/facades/resolve/main/data/train-00000-of-00001.parquet'

def fetch(output, count):
    import pyarrow.parquet as pq
    output.mkdir(parents=True, exist_ok=True)
    parquet = output / 'source.parquet'
    if not parquet.exists():
        temporary = output / 'source.parquet.partial'
        urllib.request.urlretrieve(SOURCE, temporary)
        temporary.rename(parquet)
    table = pq.read_table(parquet)
    if count > len(table):
        raise ValueError(f'Only {len(table)} upstream rows available')
    records=[]
    for i,row in enumerate(table.slice(0,count).to_pylist()):
        record={'id':i}
        for key,column in [('sketch','imageA'),('art','imageB')]:
            data=row[column]['bytes'];target=output/f'{i:06d}_{key}.png';target.write_bytes(data)
            record[key]=target.name;record[key+'_sha256']=hashlib.sha256(data).hexdigest()
        records.append(record)
    (output/'manifest.json').write_text(json.dumps({'dataset':'huggan/facades','source':SOURCE,'source_sha256':hashlib.sha256(parquet.read_bytes()).hexdigest(),'selection':f'first {count} paired rows; exploratory resplit of upstream training data','pairs':records},indent=2))
    print(f'Extracted {len(records)} real paired examples')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=256);args=p.parse_args()
    if not 10<=args.count<=400:p.error('Use 10–400 pairs')
    fetch(args.output,args.count)
