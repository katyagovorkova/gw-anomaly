from config import (
    VERSION,
    PERIOD,
    DATA_LOCATION,
    MODELS_LOCATION,
    FM_LOCATION,
    TIMESLIDES_START,
    TIMESLIDES_STOP,
    TIMESLIDE_TOTAL_DURATION
    )

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
    output:
        save_path = 'output/{period}_intersections.npy'
    script:
        'scripts/segments_intersection.py'

rule run_omicron:
    params:
        user_name = 'katya.govorkova',
        folder = f'output/omicron/',
        intersections = f'output/{PERIOD}_intersections.npy'
    shell:
        'mkdir -p {params.folder}; '
        'ligo-proxy-init {params.user_name}; '
        'python3 scripts/run_omicron.py {params.intersections} {params.folder}'

rule fetch_site_data:
    input:
        omicron = rules.run_omicron.params.folder,
        intersections = expand(rules.find_valid_segments.output.save_path,
            period=PERIOD)
    output:
        'tmp/dummy_{version}_{site}.txt'
    shell:
        'touch {output}; '
        'python3 scripts/fetch_data.py {input.omicron} {input.intersections}\
            --site {wildcards.site}'

rule fetch_timeslide_data:
    """
    O3a
    # 1238166018 -- 1 april 2019
    # 1243382418 -- 1 june 2019
    # 1248652818 -- 1 august 2019
    # 1253977218 -- 1 oct 2019
    """
    params:
        start = 1256663958,
        stop = 1257663958
    shell:
        'python3 scripts/fetch_timeslide_data.py {params.start} {params.stop}'

rule generate_data:
    input:
        omicron = '/home/katya.govorkova/gw-anomaly/output/omicron/',
        intersections = expand(rules.find_valid_segments.output.save_path,
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

rule train_gwak:
    input:
        data = expand('{datalocation}/{dataclass}.npz',
                      dataclass='{dataclass}',
                      datalocation=DATA_LOCATION),
    output:
        savedir = directory('output/{version}/trained/{dataclass}'),
        model_file = 'output/{version}/trained/models/{dataclass}.pt'
    shell:
        'mkdir -p {output.savedir}; '
        'python3 scripts/train_gwak.py {input.data} {output.model_file} {output.savedir} '

rule generate_timeslides_for_far:
    input:
        model_path = expand(rules.train_gwak.output.model_file,
            dataclass=modelclasses,
            version=VERSION),
        data_path = f'output/{VERSION}/{TIMESLIDES_START}_{TIMESLIDES_STOP}/',
    params:
        from_saved_models = False,
    output:
        save_evals_path = directory(f'output/{VERSION}/{TIMESLIDES_START}_{TIMESLIDES_STOP}_'+'timeslides_GPU{id}_duration{timeslide_total_duration}_files{files_to_eval}/'),
        log_file = f'output/{VERSION}/{TIMESLIDES_START}_{TIMESLIDES_STOP}_'+'GPU{id}_duration{timeslide_total_duration}_files{files_to_eval}.log'
    shell:
        'mkdir -p {output.save_evals_path}; '
        'python3 scripts/evaluate_timeslides.py {input.model_path} {params.from_saved_models} \
            --data-path {input.data_path} \
            --save-evals-path {output.save_evals_path} \
            --files-to-eval {wildcards.files_to_eval} \
            --timeslide-total-duration {wildcards.timeslide_total_duration} \
            --gpu {wildcards.id} \
            > {output.log_file}'

rule all_timeslides_for_far:
    input:
        expand(rules.generate_timeslides_for_far.output,
            id=range(4),
            files_to_eval=-1,
            timeslide_total_duration=9460800) # 3600*24*365*1.2/4

rule evaluate_signals:
    input:
        model_path = expand(rules.train_gwak.output.model_file,
                            dataclass=modelclasses,
                            version='{version}'),
        source_file = expand('{datalocation}/{dataclass}.npz',
                             dataclass='{signal_dataclass}',
                             datalocation=DATA_LOCATION),
    params:
        from_saved_models = False,
    output:
        save_file = 'output/{version}/evaluated/{signal_dataclass}_evals.npy',
    shell:
        'python3 scripts/evaluate_data.py {input.source_file} {output.save_file} {input.model_path} {params.from_saved_models}'

rule plot_cut_efficiency:
    input:
        evaluated_data_file = expand(rules.evaluate_signals.output,
                             version='{version}',
                             signal_dataclass='{signal_dataclass}'),
    params:
        generated_data_file = expand('{datalocation}/{dataclass}.npz',
                             dataclass='{signal_dataclass}',
                             datalocation=DATA_LOCATION),
    output:
        save_file = directory('output/{version}/cut_effic/{signal_dataclass}/')
    shell:
        'python3 scripts/plot_cut_efficiency.py \
                    {params.evaluated_data_file} {output.save_file} \
                    {params.generated_data_file}'

rule generate_timeslides_for_fm:
    input:
        data_path = expand('{datalocation}/{dataclass}.npz',
            dataclass='timeslides',
            datalocation=DATA_LOCATION),
        model_path = expand(rules.train_gwak.output.model_file,
            dataclass=modelclasses,
            version=VERSION),
    params:
        from_saved_models = False,
        shorten_timeslides = True,
    output:
        save_path = directory(f'output/{VERSION}/timeslides/'),
        save_evals_path = directory(f'output/{VERSION}/timeslides/evals/'),
        save_normalizations_path = directory(f'output/{VERSION}/timeslides/normalization/'),
    shell:
        'mkdir -p {output.save_path}; '
        'mkdir -p {output.save_evals_path}; '
        'mkdir -p {output.save_normalizations_path}; '
        'python3 scripts/compute_far.py {output.save_path} {input.model_path} {params.from_saved_models} \
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
        fm_model_path = rules.train_final_metric.output.fm_model_path,
    params:
        models = expand(rules.train_gwak.output.model_file,
                        dataclass=modelclasses,
                        version=VERSION),
        from_saved_models = False,
        from_saved_fm_model = False,
        test_path = expand('{datalocation}/{dataclass}.npz',
                           dataclass='bbh',
                           datalocation=DATA_LOCATION),
        savedir = f'output/{VERSION}/paper/'
    shell:
        'mkdir -p {params.savedir}; '
        'python3 scripts/rec_and_gwak_plots.py {params.test_path} {params.models} {params.from_saved_models} \
            {input.fm_model_path} {params.from_saved_fm_model} {params.savedir}'

rule compute_far:
    input:
        data_path = expand(rules.generate_timeslides_for_far.output.save_evals_path,
            id='{far_id}',
            version=VERSION,
            timeslide_total_duration=TIMESLIDE_TOTAL_DURATION,
            files_to_eval=-1),
        metric_coefs_path = rules.train_final_metric.output.params_file,
        norm_factors_path = rules.train_final_metric.output.norm_factor_file,
        fm_model_path = rules.train_final_metric.output.fm_model_path,
        model_path = expand(rules.train_gwak.output.model_file,
            dataclass=modelclasses,
            version=VERSION),
    params:
        from_saved_models = False,
        from_saved_fm_model = False,
        shorten_timeslides = False,
    output:
        save_path = 'output/{version}/far_bins_{far_id}.npy'
    shell:
        'touch {output.save_path};'
        'python3 scripts/compute_far.py {output.save_path} {input.model_path} {params.from_saved_models} \
            --data-path {input.data_path} \
            --fm-model-path {input.fm_model_path} \
            --from-saved-fm-model {params.from_saved_fm_model} \
            --metric-coefs-path {input.metric_coefs_path} \
            --norm-factor-path {input.norm_factors_path} \
            --fm-shortened-timeslides {params.shorten_timeslides} \
            --gpu {wildcards.far_id}'

rule merge_far_hist:
    input:
        inputs = expand(rules.compute_far.output.save_path,
            far_id=[0,1,2,3],
            version=VERSION),
    output:
        save_path = f'output/{VERSION}/far_bins.npy'
    script:
        'scripts/merge_far_hist.py'

rule quak_plotting_prediction_and_recreation:
    params:
        model_path = expand(rules.train_gwak.output.model_file,
                            dataclass=modelclasses,
                            version=VERSION),
        test_data = expand('{datalocation}/{dataclass}.npz',
                           dataclass='{dataclass}',
                           datalocation=DATA_LOCATION),
        from_saved_models = False,
        reduce_loss = False,
        save_file = 'output/{VERSION}/evaluated/quak_{dataclass}.npz'
    shell:
        'python3 scripts/gwak_predict.py {input.test_data} {params.save_file} {params.reduce_loss} \
            --model-path {input.model_path} \
            --from-saved-models {params.from_saved_models} '

rule plot_results:
    input:
        dependencies = [rules.merge_far_hist.output.save_path,
            expand(rules.evaluate_signals.output.save_file,
                signal_dataclass=fm_training_classes,
                version=VERSION)],
        fm_model_path = f'output/{VERSION}/trained/fm_model.pt'
    params:
        evaluation_dir = f'output/{VERSION}/',
        save_path = directory(f'output/{VERSION}/paper/')
    shell:
        'mkdir -p {params.save_path}; '
        'python3 scripts/plotting.py {params.evaluation_dir} {params.save_path} \
            {input.fm_model_path}'

rule supervised_bbh:
    input:
        bbh = expand('/home/katya.govorkova/gwak-paper-final-models/data/{dataclass}.npz',
            dataclass='bbh',
            version=VERSION),
        timeslides = expand('/home/katya.govorkova/gwak-paper-final-models/data/{dataclass}.npz',
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