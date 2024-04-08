from config import (
    VERSION,
    TIMESLIDE_TOTAL_DURATION,
    TIMESLIDES_START,
    TIMESLIDES_STOP,
    DATA_LOCATION
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

wildcard_constraints:
    modelclass = '|'.join([x for x in modelclasses]),

rule train_gwak:
    params:
        data = expand('{datalocation}/{dataclass}.npz',
                      dataclass='{dataclass}',
                      datalocation=DATA_LOCATION),
        savedir = directory('output/{version}/trained/{dataclass}'),
        model_file = 'output/{version}/trained/models/{dataclass}.pt'
    shell:
        'mkdir -p {output.savedir}; '
        'python3 scripts/train_gwak.py {params.data} {output.model_file} {output.savedir} '

rule generate_timeslides_for_far:
    params:
        model_path = expand(rules.train_gwak.params.model_file,
            dataclass=modelclasses,
            version=VERSION),
        from_saved_models = True,
        data_path = f'/home/katya.govorkova/gw-anomaly/output/{VERSION}/{TIMESLIDES_START}_{TIMESLIDES_STOP}/',
    output:
        save_evals_path = f'output/{VERSION}/{TIMESLIDES_START}_{TIMESLIDES_STOP}_'+'timeslides_GPU{id}_duration{timeslide_total_duration}_files{files_to_eval}/',
        log_file = f'output/{VERSION}/{TIMESLIDES_START}_{TIMESLIDES_STOP}_'+'GPU{id}_duration{timeslide_total_duration}_files{files_to_eval}.log'
    shell:
        'mkdir -p {output.save_evals_path}; '
        'python3 scripts/evaluate_timeslides.py {params.model_path} {params.from_saved_models} \
            --data-path {params.data_path} \
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
            timeslide_total_duration=32875) # 3.156e+8/800/4/3

rule evaluate_signals:
    params:
        model_path = expand(rules.train_gwak.params.model_file,
                            dataclass=modelclasses,
                            version='{version}'),
        from_saved_models = True,
        source_file = expand('{datalocation}/{dataclass}.npz',
                             dataclass='{signal_dataclass}',
                             datalocation=DATA_LOCATION),
    output:
        save_file = 'output/{version}/evaluated/{signal_dataclass}_evals.npy',
    shell:
        'python3 scripts/evaluate_data.py {params.source_file} {output.save_file} {params.model_path} {params.from_saved_models}'

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
    params:
        model_path = expand(rules.train_gwak.params.model_file,
            dataclass=modelclasses,
            version=VERSION),
        from_saved_models = True,
        data_path = expand('{datalocation}/{dataclass}.npz',
            dataclass='timeslides',
            datalocation=DATA_LOCATION),
        shorten_timeslides = True,
        save_path = f'output/{VERSION}/timeslides/',
    output:
        save_evals_path = directory(f'output/{VERSION}/timeslides/evals/'),
        save_normalizations_path = directory(f'output/{VERSION}/timeslides/normalization/'),
    shell:
        'mkdir -p {params.save_path}; '
        'mkdir -p {output.save_evals_path}; '
        'mkdir -p {output.save_normalizations_path}; '
        'python3 scripts/compute_far.py {params.save_path} {params.model_path} {params.from_saved_models} \
            --data-path {params.data_path} \
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
        fm_model_path = rules.train_final_metric.params.fm_model_path,
    params:
        models = expand(rules.train_gwak.params.model_file,
                        dataclass=modelclasses,
                        version=VERSION),
        from_saved_models = True,
        from_saved_fm_model = True,
        test_path = expand('{datalocation}/{dataclass}.npz',
                           dataclass='bbh',
                           datalocation=DATA_LOCATION),
        savedir = f'output/{VERSION}/paper/'
    shell:
        'mkdir -p {params.savedir}; '
        'python3 scripts/rec_and_gwak_plots.py {params.test_path} {params.models} {params.from_saved_models} \
            {input.fm_model_path} {params.from_saved_fm_model} {params.savedir}'

rule compute_far:
    params:
        metric_coefs_path = rules.train_final_metric.params.params_file,
        norm_factors_path = rules.train_final_metric.params.norm_factor_file,
        fm_model_path = rules.train_final_metric.params.fm_model_path,
        model_path = expand(rules.train_gwak.params.model_file,
            dataclass=modelclasses,
            version=VERSION),
        from_saved_models = True,
        from_saved_fm_model = True,
        data_path = expand(rules.generate_timeslides_for_far.output.save_evals_path,
            id='{far_id}',
            version='O3av2',
            timeslide_total_duration=TIMESLIDE_TOTAL_DURATION,
            files_to_eval=-1),
        shorten_timeslides = False,
    output:
        save_path = 'output/{version}/far_bins_{far_id}.npy'
    shell:
        'python3 scripts/compute_far.py {output.save_path} {params.model_path} {params.from_saved_models} \
            --data-path {params.data_path} \
            --fm-model-path {params.fm_model_path} \
            --from-saved-fm-model {params.from_saved_fm_model} \
            --metric-coefs-path {params.metric_coefs_path} \
            --norm-factor-path {params.norm_factors_path} \
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
        model_path = expand(rules.train_gwak.params.model_file,
                            dataclass=modelclasses,
                            version=VERSION),
        test_data = expand('{datalocation}/{dataclass}.npz',
                           dataclass='{dataclass}',
                           datalocation=DATA_LOCATION),
        from_saved_models = True,
        reduce_loss = False,
        save_file = 'output/{VERSION}/evaluated/quak_{dataclass}.npz'
    shell:
        'python3 scripts/gwak_predict.py {input.test_data} {params.save_file} {params.reduce_loss} \
            --model-path {input.model_path} \
            --from-saved-models {params.from_saved_models} '

rule plot_results:
    input:
        dependencies = [ #rules.merge_far_hist.output.save_path,
            expand(rules.evaluate_signals.output.save_file,
                signal_dataclass=fm_training_classes,
                version=VERSION)],
        fm_model_path = 'output/O3av2_non_linear_bbh_only/trained/fm_model.pt'
    params:
        evaluation_dir = 'output/O3av2_non_linear_bbh_only/',
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