from config import VERSION, TIMESLIDE_TOTAL_DURATION, TIMESLIDES_START, TIMESLIDES_STOP

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

wildcard_constraints:
    modelclass = '|'.join([x for x in modelclasses]),

rule train_gwak:
    params:
        data = expand('/home/katya.govorkova/gwak-paper-final-models/data/{dataclass}.npz',
                      dataclass='{dataclass}',
                      version=VERSION),
        savedir = directory('output/{version}/trained/{dataclass}'),
        model_file = 'output/{version}/trained/models/{dataclass}.pt'
    shell:
        'mkdir -p {params.savedir}; '
        'python3 scripts/train_gwak.py {params.data} {params.model_file} {params.savedir} '

rule generate_timeslides_for_far:
    input:
        model_path = expand(
            '/home/katya.govorkova/gwak-paper-final-models/trained/models/{dataclass}.pt',
            dataclass=modelclasses),
    params:
        data_path = f'/home/katya.govorkova/gw-anomaly/output/{VERSION}/{TIMESLIDES_START}_{TIMESLIDES_STOP}/',
        save_evals_path = f'output/{VERSION}/{TIMESLIDES_START}_{TIMESLIDES_STOP}_'+'timeslides_GPU{id}_duration{timeslide_total_duration}_files{files_to_eval}/',
    output:
        f'output/{VERSION}/{TIMESLIDES_START}_{TIMESLIDES_STOP}/'+'GPU{id}_duration{timeslide_total_duration}_files{files_to_eval}.log'
    shell:
        'mkdir -p {params.save_evals_path}; '
        'python3 scripts/evaluate_timeslides.py {input.model_path}\
            --data-path {params.data_path} \
            --save-evals-path {params.save_evals_path} \
            --files-to-eval {wildcards.files_to_eval} \
            --timeslide-total-duration {wildcards.timeslide_total_duration} \
            --gpu {wildcards.id} \
            > {output}'

rule all_timeslides_for_far:
    input:
        expand(rules.generate_timeslides_for_far.output,
            id=range(4),
            files_to_eval=-1,
            timeslide_total_duration=32875) # 3.156e+8/800/4/3

rule evaluate_signals:
    params:
        source_file = expand('/home/katya.govorkova/gwak-paper-final-models/data/{dataclass}.npz',
                             dataclass='{signal_dataclass}',
                             version='{version}'),
        model_path = expand('/home/katya.govorkova/gwak-paper-final-models/trained/models/{dataclass}.pt',
                            dataclass=modelclasses,
                            version='{version}'),
    output:
        save_file = 'output/{version}/evaluated/{signal_dataclass}_evals.npy',
    shell:
        'python3 scripts/evaluate_data.py {params.source_file} {output.save_file} {params.model_path}'

rule plot_cut_efficiency:
    params:
        generated_data_file = expand('/home/katya.govorkova/gwak-paper-final-models/data/{dataclass}.npz',
                             dataclass='{signal_dataclass}',
                             version='{version}'),
        evaluated_data_file = expand(rules.evaluate_signals.output,
                             version='{version}',
                             signal_dataclass='{signal_dataclass}'),
    output:
        save_file = directory('output/{version}/cut_effic/{signal_dataclass}/')
    shell:
        'python3 scripts/plot_cut_efficiency.py \
                    {params.evaluated_data_file} {output.save_file} \
                    {params.generated_data_file}'

rule generate_timeslides_for_fm:
    params:
        model_path = expand('/home/katya.govorkova/gwak-paper-final-models/trained/models/{dataclass}.pt',
            dataclass=modelclasses,
            version=VERSION),
        data_path = expand('/home/katya.govorkova/gwak-paper-final-models/data/{dataclass}.npz',
            dataclass='timeslides',
            version=VERSION),
        shorten_timeslides = True,
        save_path = f'output/{VERSION}/timeslides/',
    # output:
        save_evals_path = f'output/{VERSION}/timeslides/evals/',
        save_normalizations_path = f'output/{VERSION}/timeslides/normalization/',
    shell:
        'mkdir -p {params.save_path}; '
        'mkdir -p {params.save_evals_path}; '
        'mkdir -p {params.save_normalizations_path}; '
        'python3 scripts/compute_far.py {params.save_path} {params.model_path} \
            --data-path {params.data_path} \
            --save-evals-path {params.save_evals_path} \
            --save-normalizations-path {params.save_normalizations_path} \
            --fm-shortened-timeslides {params.shorten_timeslides} '

rule train_final_metric:
    input:
        signals = expand(rules.evaluate_signals.output.save_file,
            signal_dataclass=fm_training_classes,
            version=VERSION),
        timeslides = f'output/{VERSION}/timeslides/evals/',
        normfactors = f'output/{VERSION}/timeslides/normalization/',
    # output:
    params:
        params_file = f'output/{VERSION}/trained/final_metric_params.npy',
        norm_factor_file = f'output/{VERSION}/trained/norm_factor_params.npy',
        fm_model_path = f'output/{VERSION}/trained/fm_model.pt'
    shell:
        'python3 scripts/final_metric_optimization.py {params.params_file} \
            {params.fm_model_path} {params.norm_factor_file} \
            --timeslide-path {input.timeslides} \
            --signal-path {input.signals} \
            --norm-factor-path {input.normfactors}'

rule recreation_and_quak_plots:
    input:
        fm_model_path = rules.train_final_metric.params.fm_model_path
    params:
        models = expand('/home/katya.govorkova/gwak-paper-final-models/trained/models/{dataclass}.pt',
                        dataclass=modelclasses,
                        version=VERSION),
        test_path = expand('/home/katya.govorkova/gwak-paper-final-models/data/{dataclass}.npz',
                           dataclass='bbh',
                           version=VERSION),
        savedir = directory('output/{VERSION}/paper/')
    shell:
        'mkdir -p {params.savedir}; '
        'python3 scripts/rec_and_quak_plots.py {params.test_path} {params.models} \
            {input.fm_model_path} {params.savedir}'

rule compute_far:
    input:
        metric_coefs_path = rules.train_final_metric.params.params_file,
        norm_factors_path = rules.train_final_metric.params.norm_factor_file,
        fm_model_path = rules.train_final_metric.params.fm_model_path,
        data_path = expand(rules.generate_timeslides_for_far.params.save_evals_path,
            id='{far_id}',
            version='O3av2',
            timeslide_total_duration=TIMESLIDE_TOTAL_DURATION,
            files_to_eval=-1),
    params:
        model_path = expand('/home/katya.govorkova/gwak-paper-final-models/trained/models/{dataclass}.pt',
            dataclass=modelclasses,
            version=VERSION),
        shorten_timeslides = False,
    output:
        save_path = 'output/{version}/far_bins_{far_id}.npy'
    shell:
        'python3 scripts/compute_far.py {output.save_path} {params.model_path} \
            --data-path {input.data_path} \
            --fm-model-path {input.fm_model_path} \
            --metric-coefs-path {input.metric_coefs_path} \
            --norm-factor-path {input.norm_factors_path} \
            --fm-shortened-timeslides {params.shorten_timeslides} \
            --gpu {wildcards.far_id}'

rule merge_far_hist:
    input:
        inputs = expand(rules.compute_far.output.save_path,
            far_id=[0,1,2,3],
            version=VERSION),
    params:
        save_path = f'output/{VERSION}/far_bins.npy'
    script:
        'scripts/merge_far_hist.py'

rule quak_plotting_prediction_and_recreation:
    input:
        test_data = expand('/home/katya.govorkova/gwak-paper-final-models/data/{dataclass}.npz',
                           dataclass='{dataclass}',
                           version=VERSION)
    params:
        model_path = expand('/home/katya.govorkova/gwak-paper-final-models/trained/models/{dataclass}.pt',
                            dataclass=modelclasses,
                            version=VERSION),
        reduce_loss = False,
        save_file = 'output/{VERSION}/evaluated/quak_{dataclass}.npz'
    shell:
        'python3 scripts/quak_predict.py {input.test_data} {params.save_file} {params.reduce_loss} \
            --model-path {params.model_path} '

rule plot_results:
    input:
        dependencies = [rules.merge_far_hist.params.save_path,
            expand(rules.evaluate_signals.output.save_file,
                signal_dataclass=fm_training_classes,
                version=VERSION)],
        fm_model_path = rules.train_final_metric.params.fm_model_path
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