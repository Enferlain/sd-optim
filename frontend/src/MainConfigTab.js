import React, { useState, useEffect } from 'react';
import { useForm } from 'react-hook-form';
// Remove the useState hook for the config state
const MainConfigTab = () => {
  const { register, handleSubmit, formState: { errors }, reset, watch, getValues } = useForm();
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // For Google Cloud Workstations, each port gets its own subdomain
    const currentHost = window.location.hostname;
    
    const getBackendHost = () => {
      if (currentHost.includes('-')) {
        return currentHost.replace(/^\d+-/, '8000-');
      }
      return 'localhost:8000'; // Fallback for local development
    };
    const apiUrl = `https://${getBackendHost()}/config/`;
    console.log('Fetching config from:', apiUrl); // Debug log
    fetch(apiUrl)
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        reset(data); // Use reset to populate the form with fetched data
        setIsLoading(false);
      })
      .catch(error => {
        setError(error);
        setIsLoading(false);
      });
  }, []);

  const onSubmit = (data) => {
    console.log("Form data submitted:", data);
  };

  // Adjust the conditional rendering
  return (
    <div>
      <h2>Main Configuration</h2>
      {isLoading && <p>Loading configuration...</p>}
      {error && <p>Error loading configuration: {error.message}</p>} {/* config removed here */}
      {!isLoading && !error && ( // Render the form if not loading and no error
        <form onSubmit={handleSubmit(onSubmit)}>
          <div>
            <label htmlFor="run_name">Run Name:</label>
            <input
              type="text"
              id="run_name"
              {...register('run_name')}
            />
            {/* We'll add error handling later */}
            {/* {errors.run_name && <span>This field is required</span>} */}
          </div>
          <div>
            <label htmlFor="hydra.run.dir">Hydra Run Directory:</label>
            <input
              type="text"
              id="hydra.run.dir"
              name="hydra.run.dir"
              {...register('hydra.run.dir')}
            />
          </div>
          <div>
            <label htmlFor="webui">WebUI:</label>
            <select
              id="webui"
              name="webui"
              {...register('webui')}
            >
              {/* Replace with dynamic options later */}
              <option value="a1111">A1111</option>
              <option value="forge">Forge</option>
              <option value="reforge">Reforge</option>
              <option value="comfy">Comfy</option>
              <option value="swarm">Swarm</option>
            </select>
          </div>

          {/* File Paths Section */}
          <div>
            <h3>File Paths</h3>
            <div>
              <label htmlFor="configs_dir">Configs Directory:</label>
              <input
                type="text"
                id="configs_dir"
                name="configs_dir"
              {...register('configs_dir')}
              />
            </div>
            <div>
              <label htmlFor="conversion_dir">Conversion Directory:</label>
              <input
                type="text"
                id="conversion_dir"
                name="conversion_dir"
                {...register('conversion_dir')}
              />
            </div>
            <div>
              <label htmlFor="wildcards_dir">Wildcards Directory:</label>
              <input
                type="text"
                id="wildcards_dir"
                name="wildcards_dir"
                {...register('wildcards_dir')}
              />
            </div>
            <div>
              <label htmlFor="scorer_model_dir">Scorer Model Directory:</label>
              <input
                type="text"
                id="scorer_model_dir"
                name="scorer_model_dir"
                {...register('scorer_model_dir')}
              />
            </div>
          </div>

          {/* Model Inputs Section */}
          <div>
            <h3>Model Inputs</h3>
            {/* You'll need to refactor this section to work with React Hook Form for dynamic fields */}
              <div key={index} style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
                <input
                  type="text"
                  // React Hook Form registration for dynamic arrays is a bit more complex
                  // We'll address this specific case later if needed
                  // For now, we'll leave it as it was or simplify
                  style={{ flexGrow: 1, marginRight: '10px' }}
                />
                <button
                  type="button"
                  // onClick={() => handleSetBaseModel(index)} // Adjust handler for React Hook Form
                  style={{ marginRight: '5px', backgroundColor: /* config.base_model_index === index ? 'lightblue' : '' */ '', cursor: 'pointer' }} // Adjust style based on form state
                >
                  Base
                </button>
                <button
                  type="button"
                  // onClick={() => handleSetFallbackModel(index)} // Adjust handler for React Hook Form
                  style={{ marginRight: '10px', backgroundColor: /* config.fallback_model_index === index ? 'lightblue' : '' */ '', cursor: 'pointer' }} // Adjust style based on form state
                >
                  Fallback
                </button>
                <button type="button" onClick={() => handleRemoveModelPath(index)}>Remove</button>
              </div>
            ))}
            <button type="button" onClick={handleAddModelPath}>Add Model Path</button>
          </div>
          {/* Merge Settings Section */}
          <div>
            <h3>Merge Settings</h3>
            <div>
              <label htmlFor="merge_method">Merge Method:</label>
              <select
                id="merge_method"
                name="merge_method"
                {...register('merge_method')}
              >
                <option value="">Select a merge method</option>
              </select>
            </div>
            <div>
              <label htmlFor="device">Device:</label>
              <select
                id="device"
                name="device"
                {...register('device')}
              >
                <option value="cuda">cuda</option>
                <option value="cpu">cpu</option>
              </select>
            </div>
            <div>
              <label htmlFor="threads">Threads:</label>
              <input
                type="number"
                id="threads"
                name="threads"
                {...register('threads', { valueAsNumber: true })} // Register as number
              />
            </div>
            <div>
              <label htmlFor="merge_dtype">Merge Data Type:</label>
              <select
                id="merge_dtype"
                name="merge_dtype"
                {...register('merge_dtype')}
              >
                <option value="fp16">fp16</option>
                <option value="bf16">bf16</option>
                <option value="fp32">fp32</option>
                <option value="fp64">fp64</option>
              </select>
            </div>
            <div>
              <label htmlFor="save_dtype">Save Data Type:</label>
              <select
                id="save_dtype"
                name="save_dtype"
                {...register('save_dtype')}
              >
                <option value="fp16">fp16</option>
                <option value="bf16">bf16</option>
                <option value="fp32">fp32</option>
                <option value="fp64">fp64</option>
              </select>
            </div>
          </div>

          {/* Optimization Mode Section */}
          <div>
            <h3>Optimization Mode</h3>
            <div>
              <label htmlFor="optimization_mode">Mode:</label>
              <select
                id="optimization_mode"
                name="optimization_mode"
                {...register('optimization_mode')}
              >
                <option value="merge">merge</option>
                <option value="recipe">recipe</option>
                <option value="layer_adjust">layer_adjust</option>
              </select>
            </div>
 */}
            {watch('optimization_mode') === 'recipe' && ( // Use watch to conditionally render based on form value
              <div style={{ marginLeft: '20px' }}> {/* Indent recipe settings */}
                <div>
                  <label htmlFor="recipe_optimization.recipe_path">Recipe Path:</label>
                  <input
                    type="text"
                    id="recipe_optimization.recipe_path"
                    {...register('recipe_optimization.recipe_path')}
                  />
                </div>
                <div>
                  <label htmlFor="recipe_optimization.target_nodes">Target Nodes:</label>
                  {/* TODO: Implement a more user-friendly selector with backend integration */}
                  <input
                    type="text"
                    id="recipe_optimization.target_nodes"
                    {...register('recipe_optimization.target_nodes')} // Need to handle array conversion later
                  />
                </div>
                <div>
                  <label htmlFor="recipe_optimization.target_params">Target Params:</label>
                  {/* TODO: Implement a more user-friendly selector with backend integration */}
                  <input
                    type="text"
                    id="recipe_optimization.target_params"
                    {...register('recipe_optimization.target_params')} // Need to handle array conversion later
                  />
                </div>
              </div>
            )}
          </div>

          {/* General Workflow Section */}
          <div>
            <h3>General Workflow</h3>
            <div>
              <label htmlFor="save_merge_artifacts">Save Merge Artifacts:</label>
              <input
                type="checkbox"
                id="save_merge_artifacts"
                {...register('save_merge_artifacts')}
              />
            </div>
            <div>
              <label htmlFor="save_best">Save Best:</label>
              <input
                type="checkbox"
                id="save_best"
                {...register('save_best')} />
            </div>
          </div>

          {/* Optimizer Configuration Section */}
          <div>
            <h3>Optimizer Configuration</h3>
            {/* Optimizer Selection (Dropdown) */}
            <div>
              <label>Select Optimizer:</label>
              <select
                {...register('optimizer.type')} // Register a new field to hold the selected optimizer type
              >
                <option value="">Select an optimizer</option>
                <option value="bayes">Bayes</option>
                <option value="optuna">Optuna</option>
              </select>
            </div>

            {/* Conditional rendering based on the selected optimizer type */}
            {watch('optimizer.type') === 'bayes' && ( // Use watch to conditionally render Bayes settings
              <div>
                <h4>Bayes Optimizer Settings</h4>
                <div>
                  <label htmlFor="bayes_config.load_log_file">Load Log File:</label>
                  <input
                    type="text"
                    {...register('optimizer.bayes_config.load_log_file')}
                  />
                </div>
                <div>
                  <label htmlFor="bayes_config.reset_log_file">Reset Log File:</label>
                  <input
                    type="checkbox"
                    {...register('optimizer.bayes_config.reset_log_file')}
                  />
                </div>
                <div>
                  <label htmlFor="bayes_config.sampler">Sampler:</label>
                  <input
                    type="text"
                    id="bayes_config.sampler"
                    {...register('optimizer.bayes_config.sampler')}
                  />
                </div>

                {/* Acquisition Function Settings */}
                <div>
                  <h5>Acquisition Function</h5>
                  <div>
                    <label htmlFor="bayes_config.acquisition_function.kind">Kind:</label>
                    <input
                      type="text"
                      id="bayes_config.acquisition_function.kind"
                      {...register('optimizer.bayes_config.acquisition_function.kind')}
                    />
                  </div>
                  <div>
                    <label htmlFor="bayes_config.acquisition_function.kappa">Kappa:</label>
                    <input
                      type="number"
                      id="bayes_config.acquisition_function.kappa"
                      {...register('optimizer.bayes_config.acquisition_function.kappa', { valueAsNumber: true })}
                    />
                  </div>
                  <div>
                    <label htmlFor="bayes_config.acquisition_function.xi">Xi:</label>
                    <input
                      type="number"
                      id="bayes_config.acquisition_function.xi"
                      {...register('optimizer.bayes_config.acquisition_function.xi', { valueAsNumber: true })}
                    />
                  </div>
                  <div>
                    <label htmlFor="bayes_config.acquisition_function.kappa_decay">Kappa Decay:</label>
                    <input
                      type="number"
                      id="bayes_config.acquisition_function.kappa_decay"
                      {...register('optimizer.bayes_config.acquisition_function.kappa_decay', { valueAsNumber: true })}
                    />
                  </div>
                  <div>
                    <label htmlFor="bayes_config.acquisition_function.kappa_decay_delay">Kappa Decay Delay:</label>
                    <input
                      type="text" // Can be int or string
                      id="bayes_config.acquisition_function.kappa_decay_delay"
                      {...register('optimizer.bayes_config.acquisition_function.kappa_decay_delay')}
                    />
                  </div>
                </div>

                {/* Bounds Transformer Settings */}
                <div>
                  <h5>Bounds Transformer</h5>
                  <div>
                    <label htmlFor="bayes_config.bounds_transformer.enabled">Enabled:</label>
                    <input
 type="checkbox"
                      {...register('optimizer.bayes_config.bounds_transformer.enabled')}
                    />
                  </div>
                  <div>
                    <label htmlFor="bayes_config.bounds_transformer.gamma_osc">Gamma OSC:</label>
                    <input
                      type="number"
                      {...register('optimizer.bayes_config.bounds_transformer.gamma_osc', { valueAsNumber: true })}
                    />
                  </div>
                  <div>
                    <label htmlFor="bayes_config.bounds_transformer.gamma_pan">Gamma PAN:</label>
                    <input
                      type="number"
                      {...register('optimizer.bayes_config.bounds_transformer.gamma_pan', { valueAsNumber: true })}
                    />
                  </div>
                  <div>
                    <label htmlFor="bayes_config.bounds_transformer.eta">Eta:</label>
                    <input
                      type="number"
                      {...register('optimizer.bayes_config.bounds_transformer.eta', { valueAsNumber: true })}
                    />
                  </div>
                  <div>
                    <label htmlFor="bayes_config.bounds_transformer.minimum_window">Minimum Window:</label>
                    <input
                      type="number"
                      {...register('optimizer.bayes_config.bounds_transformer.minimum_window', { valueAsNumber: true })}
                    />
                  </div>
                </div>
              </div>
            )}

            {watch('optimizer.type') === 'optuna' && ( // Use watch to conditionally render Optuna settings
              <div>
                <h4>Optuna Optimizer Settings</h4>
                <div>
                  <label htmlFor="optuna_config.storage_dir">Storage Directory:</label>
                  <input
                    type="text"
                    {...register('optimizer.optuna_config.storage_dir')}
                  />
                </div>
                <div>
                  <label htmlFor="optuna_config.resume_study_name">Resume Study Name:</label>
                  <input
                    type="text"
                    {...register('optimizer.optuna_config.resume_study_name')}
                  />
                </div>
                <div>
                  <label htmlFor="optuna_config.use_pruning">Use Pruning:</label>
                  <input
                    type="checkbox"
                    {...register('optimizer.optuna_config.use_pruning')}
                  />
                </div>
                <div>
                  <label htmlFor="optuna_config.pruner_type">Pruner Type:</label>
                  <input
                    type="text"
                    {...register('optimizer.optuna_config.pruner_type')}
                  />
                </div>
                <div>
                  <label htmlFor="optuna_config.early_stopping">Early Stopping:</label>
                  <input
                    type="checkbox"
                    {...register('optimizer.optuna_config.early_stopping')}
                  />
                </div>
                <div>
                  <label htmlFor="optuna_config.patience">Patience:</label>
                  <input
                    type="number"
                    {...register('optimizer.optuna_config.patience', { valueAsNumber: true })}
                  />
                </div>
                <div>
                  <label htmlFor="optuna_config.min_improvement">Minimum Improvement:</label>
                  <input
                    type="number"
                    {...register('optimizer.optuna_config.min_improvement', { valueAsNumber: true })}
                  />
                </div>
                <div>
                  <label htmlFor="optuna_config.n_jobs">N Jobs:</label>
                  <input
                    type="number"
                    {...register('optimizer.optuna_config.n_jobs', { valueAsNumber: true })}
                  />
                </div>
                <div>
                  <label htmlFor="optuna_config.launch_dashboard">Launch Dashboard:</label>
                  <input
                    type="checkbox"
                    {...register('optimizer.optuna_config.launch_dashboard')}
                  />
                </div>
                <div>
                  <label htmlFor="optuna_config.dashboard_port">Dashboard Port:</label>
                  <input
                    type="number"
 id="optuna_config.dashboard_port"
                    {...register('optimizer.optuna_config.dashboard_port', { valueAsNumber: true })}
                  />
                </div>

                {/* Optuna Sampler Settings */}
                <div>
                  <h5>Optuna Sampler</h5>
                  <div>
                    <label htmlFor="optuna_config.sampler.type">Type:</label>
                    <input
                      type="text"
                      {...register('optimizer.optuna_config.sampler.type')}
                    />
                  </div>
                  <div>
                    <label htmlFor="optuna_config.sampler.multivariate">Multivariate:</label>
                    <input
                      type="checkbox"
                      {...register('optimizer.optuna_config.sampler.multivariate')}
                    />
                  </div>

                  <div>
                    <label htmlFor="optuna_config.sampler.group">Group:</label>
                    <input
                      type="checkbox"
                      {...register('optimizer.optuna_config.sampler.group')}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
          {/* Add more form fields here later */} {/* Closing form tag moved */}
        </form>
      )}
    </div>
  );
};

export default MainConfigTab;
