import os,sys,json,time,traceback
from pathlib import Path
import argparse
parser = argparse.ArgumentParser(description="Execute this project's notebook against recovered real data.")
parser.add_argument("--mode", choices=["quick", "full"], default="quick")
args = parser.parse_args()
os.environ["RUN_MODE"] = args.mode
root=Path(__file__).resolve().parents[1];os.chdir(root)
sys.path.insert(0, str(root))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
os.environ['MPLBACKEND']='Agg'
os.environ['MPLCONFIGDIR']=str(root / ".cache" / "matplotlib")
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['RUN_MODE']=os.environ.get('RUN_MODE','quick')
os.environ.setdefault("NLTK_DATA", str(root / ".cache" / "nltk_data"))
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
shell=InteractiveShell.instance()
shell.run_line_magic('matplotlib','inline') if False else None
path=next(root.glob('*.ipynb'));nb=json.loads(path.read_text(encoding='utf-8'))
start=time.time();result={'repo':root.name,'mode':os.environ['RUN_MODE'],'cells_completed':0}
for i,c in enumerate(nb['cells']):
    if c['cell_type']!='code':continue
    source=''.join(c['source']).replace('%matplotlib inline','# inline plots disabled in automated execution')
    source=source.replace('plt.show()', 'plt.close("all")')
    if i%20==0:print(f'{root.name} cell {i}',flush=True)
    if os.environ.get('DEBUG_CELLS') and i>=75:
        print('BEFORE',i,{k:getattr(v,'shape',None) for k,v in shell.user_ns.items() if k in ['df','df1','X_train','X_test','y_train','y_test']},flush=True)
        print(source[:400],flush=True)
    try:
        with capture_output(stdout=not bool(os.environ.get('DEBUG_CELLS')),stderr=not bool(os.environ.get('DEBUG_CELLS'))) as output:
            execution=shell.run_cell(source,store_history=False)
    except BaseException as exc:
        result.update(status='failed',cell=i,error=repr(exc),source=source)
        break
    error=execution.error_before_exec or execution.error_in_exec
    if error:
        result.update(status='failed',cell=i,error=str(error),source=source,stdout=output.stdout[-3000:],stderr=output.stderr[-3000:]);break
    result['cells_completed']+=1
    if 'plt' in shell.user_ns:shell.user_ns['plt'].close('all')
else:result['status']='passed'
result['seconds']=round(time.time()-start,2)
result['boosted_tree_backend'] = shell.user_ns.get('BOOST_BACKEND')
result['metrics'] = {}
for key, value in shell.user_ns.items():
    if key.endswith('_results') and isinstance(value, dict):
        try:
            result['metrics'][key] = {name: float(score) for name, score in value.items()}
        except (TypeError, ValueError):
            pass
if 'df_metrics' in shell.user_ns:
    result['model_comparison'] = shell.user_ns['df_metrics'].to_dict(orient='list')
result["notebook_sha256"] = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
result["python"] = sys.version.split()[0]
target=root / "results";target.mkdir(exist_ok=True)
(target / "execution-report.json").write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['stdout','stderr','source']}),flush=True)
sys.exit(0 if result['status']=='passed' else 1)
