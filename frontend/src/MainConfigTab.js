import React, { useState, useEffect, useRef } from 'react'; // <-- I added 'useRef' right here!
import { useForm, useFieldArray, Controller } from 'react-hook-form';
import styles from './MainConfigTab.module.css';
import CustomSelect from './CustomSelect.js';
import FileInput from './FileInput.js';
import { saveAs } from 'file-saver'; // Helper for downloading files
import yaml from 'js-yaml'; // Helper for YAML parsing/dumping

// Helper component
const FormRow = ({ label, children, stacked = false }) => (
    <div className={styles.formRow} style={{ alignItems: stacked ? 'flex-start' : 'center' }}>
        <label className={styles.label} style={{ paddingTop: stacked ? '6px' : '0' }}>{label}</label>
        <div className={styles.controlWrapper}>{children}</div>
    </div>
);

const MainConfigTab = () => {
    const { register, handleSubmit, reset, watch, control, setValue, getValues } = useForm({
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
    const [allScorers, setAllScorers] = useState([]);
    const importFileRef = useRef(null);

    const getBackendHost = () => {
        const currentHost = window.location.hostname;
        if (currentHost.includes('google.com')) { return currentHost.replace(/^(https?:\/\/)?(\d+)-/, 'https://8000-'); }
        return 'http://localhost:8000';
    };

    // We update our data fetching logic.
    useEffect(() => {
      const fetchData = async () => {
          try {
              const backendHost = getBackendHost();
              // Fetch config, cargo, AND our new scorers list all at once!
              const [configRes, cargoRes, scorersRes] = await Promise.all([
                  fetch(`${backendHost}/config/`),
                  fetch(`${backendHost}/config/cargo`),
                  fetch(`${backendHost}/config/scorers`) // <-- New fetch call
              ]);

              if (!configRes.ok) throw new Error(`Config fetch failed: ${configRes.statusText}`);
              if (!cargoRes.ok) throw new Error(`Cargo fetch failed: ${cargoRes.statusText}`);
              if (!scorersRes.ok) throw new Error(`Scorers fetch failed: ${scorersRes.statusText}`);

              const configData = await configRes.json();
              const cargoData = await cargoRes.json();
              const scorersData = await scorersRes.json(); // <-- Get the scorers list

              // Use the data to set our component's state!
              setCargoFiles(cargoData);
              setAllScorers(scorersData); // <-- Set the dynamic list of scorers

              // The rest of the logic is the same...
              let optimizerType = 'optuna';
              if (configData.optimizer?.bayes) optimizerType = 'bayes';
              const modelPathsAsObjects = (configData.model_paths || []).map(path => ({ value: path }));
              const initialCargo = `cargo_${configData.webui}.yaml`;
              configData.defaults = [{ payloads: initialCargo }];
              reset({ ...configData, optimizer: { ...configData.optimizer, type: optimizerType }, model_paths: modelPathsAsObjects });

          } catch (err) { setError(err); } finally { setIsLoading(false); }
      };
      fetchData();
  }, [reset]); // Dependency array doesn't need to change

    // --- NEW: Export Handler ---
    const handleExportConfig = () => {
        const currentData = getValues(); // Get all current values from the form
        
        // Clean up the data to match the YAML structure
        const exportData = { ...currentData };
        const optimizerType = exportData.optimizer.type;
        exportData.optimizer.bayes = optimizerType === 'bayes';
        exportData.optimizer.optuna = optimizerType === 'optuna';
        delete exportData.optimizer.type;
        exportData.model_paths = (exportData.model_paths || []).map(field => field.value);
        
        // Convert the JSON object to a YAML string
        try {
            const yamlString = yaml.dump(exportData, {
                noRefs: true, // Prevents YAML anchors/aliases for cleaner output
                lineWidth: -1, // Don't wrap lines
            });
            const blob = new Blob([yamlString], { type: 'text/yaml;charset=utf-8' });
            saveAs(blob, 'config.yaml'); // Trigger download
        } catch (e) {
            console.error("Error creating YAML for export:", e);
            alert("Failed to export configuration.");
        }
    };
    
    // --- NEW: Import Handler ---
    const handleImportClick = () => {
        // Programmatically click the hidden file input
        importFileRef.current?.click();
    };

    const handleFileImport = (event) => {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const text = e.target.result;
                const importedData = yaml.load(text); // Parse the YAML file
                
                // Prepare data for the form just like we do when fetching
                let optimizerType = 'optuna';
                if (importedData.optimizer?.bayes) optimizerType = 'bayes';
                const modelPathsAsObjects = (importedData.model_paths || []).map(path => ({ value: path }));
                const initialCargo = `cargo_${importedData.webui}.yaml`;
                importedData.defaults = [{ payloads: initialCargo }];

                // Use reset to update the entire form with the new data
                reset({ ...importedData, optimizer: { ...importedData.optimizer, type: optimizerType }, model_paths: modelPathsAsObjects });
                alert("Configuration imported successfully!");
            } catch (err) {
                console.error("Error importing file:", err);
                alert("Failed to import configuration. Please check if the file is a valid YAML config.");
            }
        };
        reader.readAsText(file);
        
        // Clear the input value so we can import the same file again
        event.target.value = null;
    };

    const watchedOptimizationMode = watch('optimization_mode');
    const watchedOptimizerType = watch('optimizer.type');
    const watchedWebUI = watch('webui');
    const watchedScorers = watch('scorer_method', []);
    const watchedSamplerType = watch('optimizer.optuna_config.sampler.type'); // <-- WATCHING THE SAMPLER!
    useEffect(() => { if (watchedWebUI) { setValue('defaults.0.payloads', `cargo_${watchedWebUI}.yaml`); } }, [watchedWebUI, setValue]);
    if (isLoading) return <p>Loading configuration...</p>;
    if (error) return <p>Error loading configuration: {error.message}</p>;

    const handleAddModelClick = async () => {
      if ('showOpenFilePicker' in window) {
          try {
              // We can even specify that we only want to see .safetensors files!
              const [fileHandle] = await window.showOpenFilePicker({
                  id: 'model-path-picker',
                  types: [
                      {
                          description: 'Models',
                          accept: { 'application/octet-stream': ['.safetensors', '.ckpt', '.pt'] },
                      },
                  ],
              });
              
              // Once the user selects a file, we get its name...
              const fileName = fileHandle.name;

              // And then we append a new field to our array with that name!
              append({ value: fileName });

          } catch (err) {
              // This happens if the user cancels the file picker, so we just ignore it.
              if (err.name !== 'AbortError') {
                  console.error("Error picking file:", err);
              }
          }
      } else {
          // Fallback for older browsers or environments
          // Just add an empty row like before.
          append({ value: '' });
      }
  };

    return (
        <div className={styles.formContainer}>
          {/* --- THIS IS THE PART WE'RE CHANGING --- */}
          {/* 
            Find this old line:
            <h2 style={{ paddingBottom: 'var(--space-12)' }}>Main Configuration</h2>
          */}

          {/* And replace it with this new div block: */}
          <div className={styles.headerRow}>
              <h2>Main Configuration</h2>
              <a 
                href="https://github.com/enferlain/sd-optim/wiki" // It's a real link! To our future wiki!
                target="_blank" // This makes it open in a new tab!
                rel="noopener noreferrer" // This is for security, Onii-chan!
                className={styles.buttonOutlineSm} // Use our cute small outline button style
              >
                  Wiki
              </a>
          </div>
          
          <div className={styles.formSection}>
              <h3 className={styles.legend}>Runtime</h3>
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
              <FormRow label="Configs Directory">
                  <FileInput name="configs_dir" control={control} directory={true} placeholder="path/to/model_configs" />
              </FormRow>
              <FormRow label="Conversion Directory">
                  <FileInput name="conversion_dir" control={control} directory={true} placeholder="path/to/model_configs" />
              </FormRow>
              <FormRow label="Wildcards Directory">
                  <FileInput name="wildcards_dir" control={control} directory={true} placeholder="path/to/wildcards" />
              </FormRow>
              <FormRow label="Scorer Model Directory">
                  <FileInput name="scorer_model_dir" control={control} directory={true} placeholder="path/to/Scorer" />
              </FormRow>
          </div>

          {/* --- Model Inputs Section (New and Improved!) --- */}
          <div className={styles.formSection}>
              <h3 className={styles.legend}>Model Inputs</h3>
              {fields.map((field, index) => (
                  <div key={field.id} style={{ display: 'flex', alignItems: 'center', marginBottom: '8px', gap: '8px' }}>
                      {/* Go back to a simple input, as the "Add" button now handles picking */}
                      <input
                          {...register(`model_paths.${index}.value`)}
                          placeholder="Path to model.safetensors"
                          className={styles.input}
                      />
                      
                        {/* --- THE NEW ONCLICK LOGIC USING NULL --- */}
                        <button
                            type="button"
                            onClick={() => {
                                // If it's already selected, set it to null. Otherwise, select it.
                                const newIndex = watch('base_model_index') === index ? null : index;
                                setValue('base_model_index', newIndex);
                            }}
                            className={watch('base_model_index') === index ? styles.buttonPrimarySm : styles.buttonSecondarySm}
                        >
                            Base
                        </button>

                        <button
                            type="button"
                            onClick={() => {
                                // If it's already selected, set it to null. Otherwise, select it.
                                const newIndex = watch('fallback_model_index') === index ? null : index;
                                setValue('fallback_model_index', newIndex);
                            }}
                            className={watch('fallback_model_index') === index ? styles.buttonPrimarySm : styles.buttonSecondarySm}
                        >
                            Fallback
                        </button>
                        
                        <button type="button" onClick={() => remove(index)} className={styles.buttonOutlineSm}>
                            Remove
                        </button>
                  </div>
              ))}
              <div style={{ marginTop: '12px' }}>
                  {/* The "Add" button now calls our new, smart handler! */}
                  <button type="button" onClick={handleAddModelClick} className={styles.buttonSecondary}>
                      Add Model Path
                  </button>
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

                {/* --- THIS IS THE NEW PART! --- */}
                {watchedOptimizerType === 'bayes' && (
                    <div className={styles.subFieldset}>
                        <h4 style={{marginBottom: 'var(--space-12)'}}>Bayes Settings</h4>
                        <FormRow label="Load Log File">
                            <FileInput name="optimizer.bayes_config.load_log_file" control={control} placeholder="path/to/run.json" />
                        </FormRow>
                        <FormRow label="Reset Log File">
                            <input type="checkbox" {...register('optimizer.bayes_config.reset_log_file')} className={styles.checkbox} />
                        </FormRow>
                        <FormRow label="Sampler">
                            <Controller name="optimizer.bayes_config.sampler" control={control} render={({ field }) => (
                                <CustomSelect options={['sobol', 'random', 'latin_hypercube', 'halton']} value={field.value} onChange={field.onChange} />
                            )}/>
                        </FormRow>

                        <h5 style={{marginTop: 'var(--space-16)', marginBottom: 'var(--space-12)'}}>Acquisition Function</h5>
                        <FormRow label="Kind">
                            <Controller name="optimizer.bayes_config.acquisition_function.kind" control={control} render={({ field }) => (
                                <CustomSelect options={['ucb', 'ei', 'poi']} value={field.value} onChange={field.onChange} />
                            )}/>
                        </FormRow>
                        <FormRow label="Kappa"><input type="number" step="0.1" {...register('optimizer.bayes_config.acquisition_function.kappa', { valueAsNumber: true })} className={styles.input} /></FormRow>
                        <FormRow label="Xi"><input type="number" step="0.01" {...register('optimizer.bayes_config.acquisition_function.xi', { valueAsNumber: true })} className={styles.input} /></FormRow>
                        <FormRow label="Kappa Decay"><input type="number" step="0.01" {...register('optimizer.bayes_config.acquisition_function.kappa_decay', { valueAsNumber: true })} className={styles.input} /></FormRow>
                        <FormRow label="Kappa Decay Delay"><input type="text" {...register('optimizer.bayes_config.acquisition_function.kappa_decay_delay')} className={styles.input} /></FormRow>
                    </div>
                )}

              {watchedOptimizerType === 'optuna' && (
                <div className={styles.subFieldset}>
                  <h4 style={{marginBottom: 'var(--space-12)'}}>Optuna Settings</h4>
                  <FormRow label="Storage Dir">
                        <FileInput name="optimizer.optuna_config.storage_dir" control={control} directory={true} />
                    </FormRow>                  
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

            {/* --- Scoring Section (New Layout!) --- */}
            <div className={styles.formSection}>
                <h3 className={styles.legend}>Scoring</h3>
                <FormRow label="Scorers" stacked>
                    <div className={styles.scorerLayout}>
                        {/* --- The Manual Scorer (Handled Separately) --- */}
                        <div className={styles.manualScorer}>
                             <label key="manual" className={styles.scorerLabel}>
                                <input
                                    type="checkbox"
                                    value="manual"
                                    {...register('scorer_method')}
                                    onChange={(e) => {
                                        setValue('scorer_method', e.target.checked ? ['manual'] : []);
                                    }}
                                    className={styles.checkbox}
                                />
                                Manual
                            </label>
                        </div>

                        {/* A nice vertical separator line */}
                        <div className={styles.scorerSeparator}></div>

                        {/* --- The Automatic Scorers (Mapped from the filtered list) --- */}
                        <div className={styles.autoScorersGrid}>
                            {allScorers.filter(s => s !== 'manual').map(scorer => {
                                const manualSelected = watchedScorers.includes('manual');
                                return (
                                    <label key={scorer} className={styles.scorerLabel} style={{ opacity: manualSelected ? 0.5 : 1 }}>
                                        <input
                                            type="checkbox"
                                            value={scorer}
                                            {...register('scorer_method')}
                                            disabled={manualSelected}
                                            className={styles.checkbox}
                                        />
                                        {scorer}
                                    </label>
                                );
                            })}
                        </div>
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
                                            /* --- THIS IS THE SLIDER FIX --- */
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

         {/* --- NEW: Import/Export buttons replace the old submit button --- */}
         <div className={styles.formSection} style={{ display: 'flex', justifyContent: 'flex-end', gap: 'var(--space-12)' }}>
              
              {/* Hidden file input for the import functionality */}
              <input 
                type="file" 
                ref={importFileRef}
                style={{ display: 'none' }}
                accept=".yaml,.yml"
                onChange={handleFileImport}
              />

              <button type="button" className={styles.buttonSecondary} onClick={handleImportClick}>
                  Import Config
              </button>
              <button type="button" className={styles.buttonPrimary} onClick={handleExportConfig}>
                  Export Config
              </button>
          </div>
      </div>
    );
};

export default MainConfigTab; // <--- This is the line that's probably missing!