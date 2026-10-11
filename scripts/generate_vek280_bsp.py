"""Generate the selected-XSA standalone A72 BSP and link the real supervisor.
Invoked by build_vek280_host.py inside the installed Vitis Python environment.
No device connections, programming, measurements or target API calls.
"""
import json,os,re
from pathlib import Path
import vitis

job=json.loads(Path(os.environ['SENTRY_HOST_JOB']).read_text())
workspace=Path(job['workspace'])
if workspace.exists():raise ValueError('Preserve an existing host workspace')
client=vitis.create_client()
try:
    client.set_workspace(str(workspace))
    platform=client.create_platform_component(name='sentry_a72',hw_design=job['xsa'],os='standalone',cpu='psv_cortexa72_0',domain_name='sentry_standalone',no_boot_bsp=True)
    platform.build();platform.report()
    xpfm=client.find_platform_in_repos('sentry_a72')
    app=client.create_app_component(name='supervisor',platform=xpfm,domain='sentry_standalone',template='empty_application')
    app.import_files(from_loc=job['sources'],files=['main.c','sentry_supervisor.c','sentry_supervisor.h','deployment.h'])
    linker=app.get_ld_script();regions=linker.get_memory_regions()
    choices=[region for region in regions if 'ocm' in region['name'].lower()]
    if len(choices)!=1:raise ValueError('Need one actual selected-XSA OCM region: '+str(regions))
    ocm=choices[0]
    linker.set_stack_size('0x2000');linker.set_heap_size('0x1000')
    path=Path(linker.file_path);text=path.read_text()
    known={region['name'] for region in regions}
    def bind(match):
        if match[1] not in known:raise ValueError('Unrecognized linker destination '+match[1])
        return '> '+ocm['name']
    text=re.sub(r'>\s*(\w+)',bind,text);path.write_text(text)
    app.build()
    elf_files=list((workspace/'supervisor').rglob('*.elf'))
    if len(elf_files)!=1:raise ValueError('Expected one linked supervisor ELF: '+str(elf_files))
    result=dict(passed=True,xsa=job['xsa'],workspace=str(workspace),cpu='psv_cortexa72_0',os='standalone',ocm=ocm,elf=str(elf_files[0]),linker=str(path),board_access=False,boot_action=False,scope='Actual selected-XSA BSP and OCM-linked supervisor ELF; deployment runtime and implemented-system boot packaging separate')
    (Path(job['output'])/'host_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
finally:vitis.dispose()
