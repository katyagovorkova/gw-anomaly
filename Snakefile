from config import VERSION, PERIOD

signalclasses = ['bbh', 'sglf', 'sghf']
backgroundclasses = ['background', 'glitches']
modelclasses = signalclasses + backgroundclasses
fm_training_classes = [
    'bbh_fm_optimization',
    'sghf_fm_optimization',
    'sglf_fm_optimization',
    'supernova_fm_optimization',
    'wnbhf_fm_optimization',
    'wnblf_fm_optimization'
    ]
dataclasses = fm_training_classes+[
    'wnblf',
    'wnbhf',
    'supernova',
    'timeslides',
    'bbh_varying_snr',
    'sghf_varying_snr',
    'sglf_varying_snr',
    'wnbhf_varying_snr',
    'wnblf_varying_snr',
    'supernova_varying_snr']

wildcard_constraints:
    modelclass = '|'.join([x for x in modelclasses]),
    dataclass = '|'.join([x for x in dataclasses + modelclasses])


rule find_valid_segments:
    input:
        hanford_path = 'data/{period}_Hanford_segments.json',
        livingston_path = 'data/{period}_Livingston_segments.json'
    params:
        save_path = 'output/{period}_intersections.npy'
    script:
        'scripts/segments_intersection.py'

rule run_omicron:       
    params:
        user_name = 'katya.govorkova',
        folder = f'output/omicron/'
    shell:
        'mkdir -p {params.folder}; '
        'ligo-proxy-init {params.user_name}; '
        'python3 scripts/run_omicron.py {params.intersections} {params.folder}'

rule fetch_site_data:
    input:
        omicron = rules.run_omicron.params.folder,
        intersections = expand(rules.find_valid_segments.params.save_path,
            period=PERIOD)
    output:
        'tmp/dummy_{version}_{site}.txt'
    shell:
        'touch {output}; '
        'python3 scripts/fetch_data.py {input.omicron} {input.intersections}\
            --site {wildcards.site}'

rule generate_data:
    input:
        omicron = 'output/omicron/',
        intersections = expand(rules.find_valid_segments.params.save_path,
            period=PERIOD),
    params:
        dependencies = expand(rules.fetch_site_data.output,
                                site=['L1', 'H1'],
                                version=VERSION)
    output:
        file = 'output/{version}/data/{dataclass}.npz'
    shell:
        'python3 scripts/generate.py {input.omicron} {output.file} \
            --stype {wildcards.dataclass} \
            --intersections {input.intersections} \
            --period {PERIOD}'

rule upload_data:
    input:
        expand(rules.generate_data.output.file,
               dataclass='{dataclass}',
               version='{version}')
    output:
        '/home/katya.govorkova/gwak/{version}/data/{dataclass}.npz'
    shell:
        'mkdir -p /home/katya.govorkova/gwak/{wildcards.version}/data/; '
        'cp {input} {output}; '

rule validate_data:
    input:
        expand(rules.upload_data.output,
               dataclass=modelclasses+dataclasses,
               version=VERSION)
    shell:
        'mkdir -p data/{VERSION}/; '
        'python3 scripts/validate_data.py {input}'

rule train_quak:
    input:
        data = expand(rules.upload_data.output,
                      dataclass='{dataclass}',
                      version=VERSION),
    output:
        savedir = directory('output/{version}/trained/{dataclass}'),
        model_file = 'output/{version}/trained/models/{dataclass}.pt'
    shell:
        'mkdir -p {output.savedir}; '
        'python3 scripts/train_quak.py {input.data} {output.model_file} {output.savedir} '

rule generate_timeslides_for_far:
    input:
        data_path = expand(rules.upload_data.output,
            dataclass='timeslides',
            version='{version}'),
        model_path = expand(rules.train_quak.output.model_file,
            dataclass=modelclasses,
            version='{version}'),
    params:
        shorten_timeslides = False,
        save_path = 'output/{version}/timeslides_{id}/',
    output:
        save_evals_path = directory('output/{version}/timeslides_{id}/evals/'),
        save_normalizations_path = directory('output/{version}/timeslides_{id}/normalization/')
    shell:
        'mkdir -p {params.save_path}; '
        'mkdir -p {output.save_evals_path}; '
        'mkdir -p {output.save_normalizations_path}; '
        'python3 scripts/evaluate_timeslides.py {params.save_path} {input.model_path} \
            --data-path {input.data_path} \
            --save-evals-path {output.save_evals_path} \
            --save-normalizations-path {output.save_normalizations_path} \
            --fm-shortened-timeslides {params.shorten_timeslides} \
            --gpu {wildcards.id}'

rule evaluate_signals:
    input:
        source_file = expand(rules.upload_data.output,
                             dataclass='{signal_dataclass}',
                             version='{version}'),
        model_path = expand(rules.train_quak.output.model_file,
                            dataclass=modelclasses,
                            version='{version}'),
    output:
        save_file = 'output/{version}/evaluated/{signal_dataclass}_evals.npy',
    shell:
        'python3 scripts/evaluate_data.py {input.source_file} {output.save_file} {input.model_path}'

rule generate_timeslides_for_fm:
    input:
        model_path = expand(rules.train_quak.output.model_file,
            dataclass=modelclasses,
            version=VERSION),
        data_path = expand(rules.upload_data.output,
            dataclass='timeslides',
            version=VERSION),
    params:
        shorten_timeslides = True,
        save_path = f'output/{VERSION}/timeslides/',
    output:
        save_evals_path = directory(f'output/{VERSION}/timeslides/evals/'),
        save_normalizations_path = directory(f'output/{VERSION}/timeslides/normalization/'),
    shell:
        'mkdir -p {params.save_path}; '
        'mkdir -p {output.save_evals_path}; '
        'mkdir -p {output.save_normalizations_path}; '
        'python3 scripts/evaluate_timeslides.py {params.save_path} {input.model_path} \
            --data-path {input.data_path} \
            --save-evals-path {output.save_evals_path} \
            --save-normalizations-path {output.save_normalizations_path} \
            --fm-shortened-timeslides {params.shorten_timeslides} '

rule train_final_metric:
    input:
        signals = expand(rules.evaluate_signals.output.save_file,
            signal_dataclass=fm_training_classes,
            version=VERSION),
        timeslides = f'output/{VERSION}/timeslides/evals/',
        normfactors = f'output/{VERSION}/timeslides/normalization/',
    output:
        params_file = f'output/{VERSION}/trained/final_metric_params.npy',
        norm_factor_file = f'output/{VERSION}/trained/norm_factor_params.npy',
        fm_model_path = f'output/{VERSION}/trained/fm_model.pt'
    shell:
        'python3 scripts/final_metric_optimization.py {output.params_file} \
            {output.fm_model_path} {output.norm_factor_file} \
            --timeslide-path {input.timeslides} \
            --signal-path {input.signals} \
            --norm-factor-path {input.normfactors}'

rule recreation_and_quak_plots:
    input:
        fm_model_path = rules.train_final_metric.output.fm_model_path
    params:
        models = expand(rules.train_quak.output.model_file,
                        dataclass=modelclasses,
                        version=VERSION),
        test_path = expand(rules.upload_data.output,
                           dataclass='bbh',
                           version=VERSION),
        savedir = directory('output/{VERSION}/paper/')
    shell:
        'mkdir -p {params.savedir}; '
        'python3 scripts/rec_and_quak_plots.py {params.test_path} {params.models} \
            {input.fm_model_path} {params.savedir}'

rule compute_far:
    input:
        metric_coefs_path = rules.train_final_metric.output.params_file,
        norm_factors_path = rules.train_final_metric.output.norm_factor_file,
        fm_model_path = rules.train_final_metric.output.fm_model_path,
        data_path = expand(rules.generate_timeslides_for_far.output.save_evals_path,
            id='{far_id}',
            version=VERSION),
    params:
        model_path = expand(rules.train_quak.output.model_file,
            dataclass=modelclasses,
            version=VERSION),
        shorten_timeslides = False,
    output:
        save_path = 'output/{version}/far_bins_{far_id}.npy'
    shell:
        'python3 scripts/evaluate_timeslides.py {output.save_path} {params.model_path} \
            --data-path {input.data_path} \
            --fm-model-path {input.fm_model_path} \
            --metric-coefs-path {input.metric_coefs_path} \
            --norm-factor-path {input.norm_factors_path} \
            --fm-shortened-timeslides {params.shorten_timeslides} \
            --gpu {wildcards.far_id}'

rule merge_far_hist:
    params:
        inputs = expand(rules.compute_far.output.save_path,
            far_id=[0,1,2,3],
            version=VERSION),
    output:
        save_path = f'output/{VERSION}/far_bins.npy'
    script:
        'scripts/merge_far_hist.py'

rule quak_plotting_prediction_and_recreation:
    input:
        test_data = expand(rules.upload_data.output,
                           dataclass='{dataclass}',
                           version=VERSION)
    params:
        model_path = expand(rules.train_quak.output.model_file,
                            dataclass=modelclasses,
                            version=VERSION),
        reduce_loss = False,
        save_file = 'output/{VERSION}/evaluated/quak_{dataclass}.npz'
    shell:
        'python3 scripts/quak_predict.py {input.test_data} {params.save_file} {params.reduce_loss} \
            --model-path {params.model_path} '

rule plot_results:
    input:
        dependencies = [rules.merge_far_hist.output.save_path,
            expand(rules.evaluate_signals.output.save_file,
                signal_dataclass=fm_training_classes,
                version=VERSION)],
        fm_model_path = rules.train_final_metric.output.fm_model_path
    params:
        evaluation_dir = f'output/{VERSION}/',
        save_path = directory(f'output/{VERSION}/paper/')
    shell:
        'mkdir -p {params.save_path}; '
        'python3 scripts/plotting.py {params.evaluation_dir} {params.save_path} \
            {input.fm_model_path}'

rule supervised_bbh:
    input:
        bbh = expand(rules.upload_data.output,
            dataclass='bbh',
            version=VERSION),
        timeslides = expand(rules.upload_data.output,
            dataclass='timeslides',
            version=VERSION),
    params:
        models = 'output/supervised-bbh/model.pt',
        plots = 'output/supervised-bbh/'
    shell:
        'python3 scripts/supervised.py {input.bbh} {input.timeslides} \
            {params.models} {params.plots}'

rule make_pipeline_plot:
    shell:
        'snakemake plot_results --dag | dot -Tpdf > dag.pdf'