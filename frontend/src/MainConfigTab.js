import React, { useState, useEffect } from 'react';
import { useForm, useFieldArray, Controller } from 'react-hook-form';
import styles from './MainConfigTab.module.css';
import CustomSelect from './CustomSelect.js'; // <-- Import our custom component!

// Helper component
const FormRow = ({ label, children, stacked = false }) => (
    <div className={styles.formRow} style={{ alignItems: stacked ? 'flex-start' : 'center' }}>
        <label className={styles.label} style={{ paddingTop: stacked ? '6px' : '0' }}>{label}</label>
        <div className={styles.controlWrapper}>{children}</div>
    </div>
);

const MainConfigTab = () => {
    const { register, handleSubmit, reset, watch, control, setValue } = useForm({
        // THIS PART NEEDS TO BE UPDATED!
        defaultValues: {
            defaults: [{ payloads: 'cargo_forge.yaml' }],
            run_name: '',
            hydra: { run: { dir: '' } },
            webui: 'forge',
            configs_dir: '',
            conversion_dir: '',
            wildcards_dir: '',
            scorer_model_dir: '',
            model_paths: [],
            base_model_index: 0,
            fallback_model_index: -1,
            merge_method: '',
            device: 'cuda',
            threads: 4,
            merge_dtype: 'fp32',
            save_dtype: 'bf16',
            add_extra_keys: false, // <-- New
            save_merge_artifacts: true,
            save_best: true,
            optimization_mode: 'merge',
            recipe_optimization: { recipe_path: '', target_nodes: '', target_params: '' },
            optimizer: { 
                type: 'optuna', 
                bayes: false, 
                optuna: true, 
                random_state: -1, 
                init_points: 10, 
                n_iters: 20,
                bayes_config: { 
                    load_log_file: null, 
                    reset_log_file: false, 
                    sampler: 'sobol', 
                    acquisition_function: { kind: 'ucb', kappa: 3.0, xi: 0.05, kappa_decay: 0.98, kappa_decay_delay: '${optimizer.init_points}' }, 
                    bounds_transformer: { enabled: false, gamma_osc: 0.7, gamma_pan: 1.0, eta: 0.9, minimum_window: 0.0 } 
                },
                optuna_config: { 
                    storage_dir: 'optuna_db', 
                    resume_study_name: null, 
                    use_pruning: false, 
                    pruner_type: 'median', 
                    early_stopping: false, 
                    patience: 10, 
                    min_improvement: 0.001, 
                    n_jobs: 1, 
                    sampler: { type: 'tpe', multivariate: true, group: true }, 
                    launch_dashboard: true, 
                    dashboard_port: 8080 
                }
            },
            batch_size: 1,
            save_imgs: true,
            img_average_type: 'arithmetic',
            background_check: { enabled: false, payloads: [] },
            generator_concurrency_limit: 10,
            generator_keepalive_interval: 60,
            generator_total_timeout: 0,
            scorer_method: [],
            scorer_average_type: 'arithmetic',
            scorer_weight: {},
            scorer_default_device: 'cpu',
            scorer_device: {},
            scorer_print_individual: true,
        }
    });

    const { fields, append, remove } = useFieldArray({ control, name: "model_paths" });
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const [cargoFiles, setCargoFiles] = useState([]);
    const [allScorers, setAllScorers] = useState(['laion', 'hpsv21', 'pick', 'imagereward', 'cityaes', 'manual']);

    const getBackendHost = () => {
        const currentHost = window.location.hostname;
        if (currentHost.includes('google.com')) { return currentHost.replace(/^(https?:\/\/)?(\d+)-/, 'https://8000-'); }
        return 'http://localhost:8000';
    };

    useEffect(() => {
        const fetchData = async () => {
            try {
                const backendHost = getBackendHost();
                const [configRes, cargoRes] = await Promise.all([
                    fetch(`${backendHost}/config/`),
                    fetch(`${backendHost}/config/cargo`)
                ]);
                if (!configRes.ok) throw new Error(`Config fetch failed: ${configRes.statusText}`);
                if (!cargoRes.ok) throw new Error(`Cargo fetch failed: ${cargoRes.statusText}`);
                const configData = await configRes.json();
                const cargoData = await cargoRes.json();
                setCargoFiles(cargoData);
                let optimizerType = 'optuna';
                if (configData.optimizer?.bayes) optimizerType = 'bayes';
                const modelPathsAsObjects = (configData.model_paths || []).map(path => ({ value: path }));
                const initialCargo = `cargo_${configData.webui}.yaml`;
                configData.defaults = [{ payloads: initialCargo }];
                reset({ ...configData, optimizer: { ...configData.optimizer, type: optimizerType }, model_paths: modelPathsAsObjects });
            } catch (err) { setError(err); } finally { setIsLoading(false); }
        };
        fetchData();
    }, [reset]);

    const onSubmit = (data) => {
        const submissionData = { ...data };
        const optimizerType = submissionData.optimizer.type;
        submissionData.optimizer.bayes = optimizerType === 'bayes';
        submissionData.optimizer.optuna = optimizerType === 'optuna';
        delete submissionData.optimizer.type;
        submissionData.model_paths = (submissionData.model_paths || []).map(field => field.value);
        console.log("Submitting this data to the backend:", JSON.stringify(submissionData, null, 2));
    };

    const watchedOptimizationMode = watch('optimization_mode');
    const watchedOptimizerType = watch('optimizer.type');
    const watchedWebUI = watch('webui');
    const watchedScorers = watch('scorer_method', []);
    const watchedSamplerType = watch('optimizer.optuna_config.sampler.type'); // <-- WATCHING THE SAMPLER!
    useEffect(() => { if (watchedWebUI) { setValue('defaults.0.payloads', `cargo_${watchedWebUI}.yaml`); } }, [watchedWebUI, setValue]);
    if (isLoading) return <p>Loading configuration...</p>;
    if (error) return <p>Error loading configuration: {error.message}</p>;

    return (
      <form onSubmit={handleSubmit(onSubmit)} className={styles.formContainer}>
          <h2 style={{ paddingBottom: 'var(--space-12)' }}>Main Configuration</h2>
          
          <div className={styles.formSection}>
              <h3 className={styles.legend}>Run & Payloads</h3>
              <FormRow label="Run Name">
                  <input type="text" {...register('run_name')} className={styles.input} />
              </FormRow>
              <FormRow label="WebUI">
                  <Controller name="webui" control={control} render={({ field }) => (<CustomSelect options={['forge', 'a1111', 'reforge', 'comfy', 'swarm']} value={field.value} onChange={field.onChange} />)} />
              </FormRow>
              <FormRow label="Cargo File">
                  <Controller name="defaults.0.payloads" control={control} render={({ field }) => (<CustomSelect options={cargoFiles} value={field.value} onChange={field.onChange} />)} />
              </FormRow>
          </div>

          <div className={styles.formSection}>
              <h3 className={styles.legend}>File Paths</h3>
              <FormRow label="Hydra Run Directory"><input type="text" {...register('hydra.run.dir')} className={styles.input} /></FormRow>
              <FormRow label="Configs Directory"><input type="text" {...register('configs_dir')} className={styles.input} /></FormRow>
              <FormRow label="Conversion Directory"><input type="text" {...register('conversion_dir')} className={styles.input} /></FormRow>
              <FormRow label="Wildcards Directory"><input type="text" {...register('wildcards_dir')} className={styles.input} /></FormRow>
              <FormRow label="Scorer Model Directory"><input type="text" {...register('scorer_model_dir')} className={styles.input} /></FormRow>
          </div>

          <div className={styles.formSection}>
              <h3 className={styles.legend}>Model Inputs</h3>
              {fields.map((field, index) => (
                  <div key={field.id} style={{ display: 'flex', alignItems: 'center', marginBottom: '8px', gap: '8px' }}>
                      <input {...register(`model_paths.${index}.value`)} placeholder={`Model Path ${index + 1}`} className={styles.input} />
                      <button type="button" onClick={() => setValue('base_model_index', index)} className={watch('base_model_index') === index ? styles.buttonPrimarySm : styles.buttonSecondarySm}>Base</button>
                      <button type="button" onClick={() => setValue('fallback_model_index', index)} className={watch('fallback_model_index') === index ? styles.buttonPrimarySm : styles.buttonSecondarySm}>Fallback</button>
                      <button type="button" onClick={() => remove(index)} className={styles.buttonOutlineSm}>Remove</button>
                  </div>
              ))}
              <div style={{ paddingLeft: '192px', marginTop: '12px' }}>
                  <button type="button" onClick={() => append({ value: '' })} className={styles.buttonSecondary}>Add Model Path</button>
                  <button type="button" onClick={() => setValue('fallback_model_index', -1)} className={styles.buttonSecondary} style={{ marginLeft: '10px' }}>Clear Fallback</button>
              </div>
          </div>

          <div className={styles.formSection}>
              <h3 className={styles.legend}>Merge Settings</h3>
              <FormRow label="Merge Method"><input type="text" {...register('merge_method')} className={styles.input} /></FormRow>
              <FormRow label="Device"><Controller name="device" control={control} render={({ field }) => (<CustomSelect options={['cuda', 'cpu']} value={field.value} onChange={field.onChange} />)} /></FormRow>
              <FormRow label="Threads"><input type="number" {...register('threads', { valueAsNumber: true })} className={styles.input} /></FormRow>
              <FormRow label="Merge Precision"><Controller name="merge_dtype" control={control} render={({ field }) => (<CustomSelect options={['fp16', 'bf16', 'fp32', 'fp64']} value={field.value} onChange={field.onChange} />)} /></FormRow>
              <FormRow label="Save Precision"><Controller name="save_dtype" control={control} render={({ field }) => (<CustomSelect options={['fp16', 'bf16', 'fp32', 'fp64']} value={field.value} onChange={field.onChange} />)} /></FormRow>
              <FormRow label="Add Extra Keys (v-pred)"><input type="checkbox" {...register('add_extra_keys')} className={styles.checkbox} /></FormRow>
          </div>

          <div className={styles.formSection}>
              <h3 className={styles.legend}>General Workflow</h3>
              <FormRow label="Save Merge Artifacts"><input type="checkbox" {...register('save_merge_artifacts')} className={styles.checkbox} /></FormRow>
              <FormRow label="Save Best Model"><input type="checkbox" {...register('save_best')} className={styles.checkbox} /></FormRow>
          </div>
          
          <div className={styles.formSection}>
               <h3 className={styles.legend}>Optimization Mode</h3>
              <FormRow label="Mode"><Controller name="optimization_mode" control={control} render={({ field }) => (<CustomSelect options={['merge', 'recipe', 'layer_adjust']} value={field.value} onChange={field.onChange} />)} /></FormRow>
              {watchedOptimizationMode === 'recipe' && (
                  <div className={styles.subFieldset}>
                      <h4 style={{marginBottom: 'var(--space-12)'}}>Recipe Settings</h4>
                      <FormRow label="Recipe Path"><input type="text" {...register('recipe_optimization.recipe_path')} className={styles.input} /></FormRow>
                      <FormRow label="Target Nodes"><input type="text" {...register('recipe_optimization.target_nodes')} className={styles.input} /></FormRow>
                      <FormRow label="Target Params"><input type="text" {...register('recipe_optimization.target_params')} className={styles.input} /></FormRow>
                  </div>
              )}
          </div>

          {/* --- Optimizer Configuration Section --- */}
          <div className={styles.formSection}>
              <h3 className={styles.legend}>Optimizer Configuration</h3>
              <FormRow label="Random State"><input type="number" {...register('optimizer.random_state', { valueAsNumber: true })} className={styles.input} /></FormRow>
              <FormRow label="Init Points"><input type="number" {...register('optimizer.init_points', { valueAsNumber: true })} className={styles.input} /></FormRow>
              <FormRow label="Iterations"><input type="number" {...register('optimizer.n_iters', { valueAsNumber: true })} className={styles.input} /></FormRow>
              <hr style={{margin: '20px 0', border: 'none', borderTop: '1px solid var(--color-border)'}} />
              <FormRow label="Select Optimizer">
                   <Controller name="optimizer.type" control={control} render={({ field }) => (<CustomSelect options={['optuna', 'bayes']} value={field.value} onChange={field.onChange} />)} />
              </FormRow>
              
              {watchedOptimizerType === 'optuna' && (
                <div className={styles.subFieldset}>
                  <h4 style={{marginBottom: 'var(--space-12)'}}>Optuna Settings</h4>
                  <FormRow label="Storage Dir"><input type="text" {...register('optimizer.optuna_config.storage_dir')} className={styles.input}/></FormRow>
                  <FormRow label="Resume Study"><input type="text" {...register('optimizer.optuna_config.resume_study_name')} className={styles.input} /></FormRow>
                  <FormRow label="Launch Dashboard"><input type="checkbox" {...register('optimizer.optuna_config.launch_dashboard')} className={styles.checkbox} /></FormRow>
                  <FormRow label="Dashboard Port"><input type="number" {...register('optimizer.optuna_config.dashboard_port', { valueAsNumber: true })} className={styles.input} /></FormRow>
                  
                  <h5 style={{marginTop: 'var(--space-16)', marginBottom: 'var(--space-12)'}}>Sampler</h5>
                  <FormRow label="Type">
                      <Controller name="optimizer.optuna_config.sampler.type" control={control} render={({ field }) => (<CustomSelect options={['tpe', 'cmaes', 'random', 'grid', 'qmc']} value={field.value} onChange={field.onChange} />)} />
                  </FormRow>
                  {watchedSamplerType === 'cmaes' && (<FormRow label="Restart Strategy"><Controller name="optimizer.optuna_config.sampler.restart_strategy" control={control} render={({ field }) => (<CustomSelect options={['', 'ipop', 'bipop']} value={field.value} onChange={field.onChange} />)} /></FormRow>)}
                  {watchedSamplerType === 'qmc' && (<FormRow label="QMC Type"><Controller name="optimizer.optuna_config.sampler.qmc_type" control={control} render={({ field }) => (<CustomSelect options={['sobol', 'halton', 'lhs']} value={field.value} onChange={field.onChange} />)} /></FormRow>)}
                  {watchedSamplerType === 'tpe' && (<> <FormRow label="Multivariate"><input type="checkbox" {...register('optimizer.optuna_config.sampler.multivariate')} className={styles.checkbox} /></FormRow> <FormRow label="Group"><input type="checkbox" {...register('optimizer.optuna_config.sampler.group')} className={styles.checkbox} /></FormRow> </>)}
                  {watchedSamplerType === 'qmc' && (<FormRow label="Scramble"><input type="checkbox" {...register('optimizer.optuna_config.sampler.scramble')} className={styles.checkbox} /></FormRow>)}
                  {watchedSamplerType === 'grid' && (<FormRow label="Search Space" stacked><textarea {...register('optimizer.optuna_config.sampler.search_space')} className={styles.textarea} rows={4} placeholder={'alpha: [0.1, 0.5]\nbeta: [10, 20]'}></textarea></FormRow>)}
                </div>
              )}
          </div>
          
          <div className={styles.formSection}>
                <h3 className={styles.legend}>Image Generation</h3>
                <FormRow label="Images per Payload"><input type="number" {...register('batch_size', { valueAsNumber: true })} className={styles.input} /></FormRow>
                <FormRow label="Save Images"><input type="checkbox" {...register('save_imgs')} className={styles.checkbox} /></FormRow>
                <FormRow label="Image Score Avg"><Controller name="img_average_type" control={control} render={({ field }) => (<CustomSelect options={['arithmetic', 'geometric', 'quadratic']} value={field.value} onChange={field.onChange} />)} /></FormRow>
                <FormRow label="Background Check"><input type="checkbox" {...register('background_check.enabled')} className={styles.checkbox} /></FormRow>
            </div>

            <div className={styles.formSection}>
                <h3 className={styles.legend}>Connection Settings</h3>
                <FormRow label="Concurrency Limit"><input type="number" {...register('generator_concurrency_limit', { valueAsNumber: true })} className={styles.input} /></FormRow>
                <FormRow label="Keep-Alive (s)"><input type="number" {...register('generator_keepalive_interval', { valueAsNumber: true })} className={styles.input} /></FormRow>
                <FormRow label="Total Timeout (s)"><input type="number" {...register('generator_total_timeout', { valueAsNumber: true })} className={styles.input} /></FormRow>
            </div>

            <div className={styles.formSection}>
                <h3 className={styles.legend}>Scoring</h3>
                <FormRow label="Scorers" stacked>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px' }}>
                        {allScorers.map(scorer => {
                            const isManual = scorer === 'manual';
                            const manualSelected = watchedScorers.includes('manual');
                            return (
                                <label key={scorer} style={{ display: 'flex', alignItems: 'center', gap: '5px', opacity: isManual ? 1 : (manualSelected ? 0.5 : 1) }}>
                                    <input type="checkbox" value={scorer} {...register('scorer_method')} disabled={!isManual && manualSelected} onChange={(e) => {
                                        const { checked, value } = e.target;
                                        if (value === 'manual' && checked) {
                                            setValue('scorer_method', ['manual']);
                                        } else {
                                            const currentScorers = watchedScorers.filter(s => s !== 'manual');
                                            if (checked) {
                                                setValue('scorer_method', [...currentScorers, value]);
                                            } else {
                                                setValue('scorer_method', currentScorers.filter(s => s !== value));
                                            }
                                        }
                                    }} className={styles.checkbox} />
                                    {scorer}
                                </label>
                            );
                        })}
                    </div>
                </FormRow>
                
                {watchedScorers.length > 0 && !watchedScorers.includes('manual') && (
                    <div className={styles.subFieldset}>
                        <FormRow label="Scorer Avg Type"><Controller name="scorer_average_type" control={control} render={({ field }) => (<CustomSelect options={['arithmetic', 'geometric', 'quadratic']} value={field.value} onChange={field.onChange} />)}/></FormRow>
                        <FormRow label="Default Scorer Device"><Controller name="scorer_default_device" control={control} render={({ field }) => (<CustomSelect options={['cpu', 'cuda']} value={field.value} onChange={field.onChange} />)}/></FormRow>
                        <FormRow label="Print Individual Scores"><input type="checkbox" {...register('scorer_print_individual')} className={styles.checkbox}/></FormRow>
                        
                        <hr style={{margin: '20px 0', border: 'none', borderTop: '1px solid var(--color-border)'}} />
                        
                        {watchedScorers.map(scorer => (
                            <div key={scorer}>
                                <h4 style={{marginLeft: '20px', marginBottom:'10px'}}>{scorer}</h4>
                                <FormRow label="Weight">
                                    <Controller
                                        name={`scorer_weight.${scorer}`}
                                        control={control}
                                        defaultValue={1.0}
                                        render={({ field }) => (
                                            <div style={{display: 'flex', alignItems: 'center', width: '100%', gap: '15px'}}>
                                                <input type="range" min="0" max="2" step="0.05" {...field} style={{flexGrow: 1}} />
                                                <span className={styles.sliderValue}>{Number(field.value).toFixed(2)}</span>
                                            </div>
                                        )}
                                    />
                                </FormRow>
                                <FormRow label="Device Override">
                                    <Controller name={`scorer_device.${scorer}`} control={control} render={({ field }) => (<CustomSelect options={['', 'cuda', 'cpu']} value={field.value} onChange={field.onChange} />)} />
                                </FormRow>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            <button type="submit" className={styles.submitButton}>Save Configuration</button>
        </form>
    );
};

export default MainConfigTab;