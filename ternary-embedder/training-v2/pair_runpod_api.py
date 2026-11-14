"""REST v2 lifecycle adapter for the data-first controller's small v1-shaped API."""
import json,stat,urllib.request,urllib.error
from pathlib import Path
from pilot_controller import RunPodError

def normalize(pod):
 if not isinstance(pod,dict):return pod
 pod=dict(pod)
 if 'cost' in pod:pod['costPerHr']=pod['cost']
 direct=(pod.get('ssh') or {}).get('direct')
 if direct:
  pod['publicIp']=direct['host'];pod['portMappings']={'22':direct['port']}
 else:
  for p in (pod.get('runtime') or {}).get('ports',[]):
   if p.get('private')==22:pod['publicIp']=p['ip'];pod['portMappings']={'22':p['public']};break
 return pod

def api(method,route,body=None):
 path=Path.home()/'.config/runpod/credentials.env'
 if stat.S_IMODE(path.stat().st_mode)!=0o600 or stat.S_IMODE(path.parent.stat().st_mode)!=0o700:raise RuntimeError('Credential permissions must be 0600/0700')
 token=next(l.split('=',1)[1].strip().strip('\"\'') for l in path.read_text().splitlines() if l.startswith('RUNPOD_API_KEY='))
 if method=='POST' and route=='pods':
  body={'name':body['name'],'image':'runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404','cloud':body['cloudType'],'gpu':{'id':body['gpuTypeIds'][0],'count':body['gpuCount'],'allowedCudaVersions':body.get('allowedCudaVersions') or ['13.2'],'minRamPerGpu':body.get('minRAMPerGPU',8),'minVcpuCountPerGpu':1},'disk':body['containerDiskInGb'],'ports':body['ports'],'env':body['env'],'startSsh':True,'dataCenterIds':body.get('dataCenterIds',[])}
 req=urllib.request.Request('https://api.runpod.io/v2/'+route,data=json.dumps(body).encode() if body is not None else None,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json','User-Agent':'TernaryDataFirst'},method=method)
 try:
  with urllib.request.urlopen(req,timeout=25) as response:
   raw=response.read();result=json.loads(raw) if raw else None
 except urllib.error.HTTPError as error:
  detail=error.read().decode(errors='replace').replace(token,'[redacted]')
  raise RunPodError(error.code,f'RunPod v2 {method} {route}: HTTP {error.code}: {detail[:1200]}') from None
 if method=='GET' and route=='pods':return [normalize(p) for p in result['pods']]
 return normalize(result)
