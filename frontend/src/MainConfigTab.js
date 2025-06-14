import React, { useState, useEffect } from 'react';

const MainConfigTab = () => {
  const [config, setConfig] = useState(null);
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
        setConfig(data);
        setIsLoading(false);
      })
      .catch(error => {
        setError(error);
        setIsLoading(false);
      });
  }, []);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    // Handle nested structure for hydra.run.dir
    if (name === 'hydra.run.dir') {
      setConfig(prevConfig => ({
        ...prevConfig,
        hydra: {
          ...prevConfig.hydra,
          run: {
            ...prevConfig.hydra.run,
            dir: value
          }
        }
      }));
    } else {
      setConfig(prevConfig => ({
        ...prevConfig,
        [name]: value
      }));
    }
  };

  const handleModelPathChange = (index, value) => {
    setConfig(prevConfig => {
      const newModelPaths = [...prevConfig.model_paths];
      newModelPaths[index] = value;
      return { ...prevConfig, model_paths: newModelPaths };
    });
  };

  const handleAddModelPath = () => {
    setConfig(prevConfig => ({
      ...prevConfig,
      model_paths: [...prevConfig.model_paths, '']
    }));
  };

  const handleRemoveModelPath = (index) => {
    setConfig(prevConfig => ({
      ...prevConfig,
      model_paths: prevConfig.model_paths.filter((_, i) => i !== index)
    }));
  };

  const handleSetBaseModel = (index) => {
    setConfig(prevConfig => ({
      ...prevConfig,
      base_model_index: index
    }));
  };

  const handleSetFallbackModel = (index) => {
    setConfig(prevConfig => ({
      ...prevConfig,
      fallback_model_index: index === prevConfig.fallback_model_index ? -1 : index // Toggle fallback
    }));
  };

  const handleMergeInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setConfig(prevConfig => ({
      ...prevConfig,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleRecipeInputChange = (field, e) => {
    const { value } = e.target;
    setConfig(prevConfig => ({
      ...prevConfig,
      recipe_optimization: {
        ...prevConfig.recipe_optimization,
        [field]: value
      }
    }));
  };

  const handleOptimizationModeChange = (e) => {
    const { value } = e.target;
    setConfig(prevConfig => ({
      ...prevConfig,
      optimization_mode: value
    }));
  };

  const handleOptimizerInputChange = (section, e) => {
    const { name, value, type, checked } = e.target;
    setConfig(prevConfig => ({
      ...prevConfig,
      optimizer: {
        ...prevConfig.optimizer,
        [section]: {
          ...prevConfig.optimizer[section],
          [name]: type === 'checkbox' ? checked : value
        }
      }
    }));
  };

  const handleNestedInputChange = (section, field, e) => {
    const { value, type, checked } = e.target;
    setConfig(prevConfig => ({
      ...prevConfig,
      [section]: {
        ...prevConfig[section],
        [field]: type === 'checkbox' ? checked : value
      }
    }));
  };

  const handleCheckboxChange = (name, checked) => {
    setConfig(prevConfig => ({
      ...prevConfig,
      [name]: checked
    }));
  };

  const handleOptimizerChange = (optimizerType) => {
    setConfig(prevConfig => {
      const newOptimizer = { ...prevConfig.optimizer };
      newOptimizer.bayes = optimizerType === 'bayes';
      newOptimizer.optuna = optimizerType === 'optuna';

      // Initialize nested optimizer config if it doesn't exist
      if (optimizerType === 'bayes' && !newOptimizer.bayes_config) {
        newOptimizer.bayes_config = {};
      }
      if (optimizerType === 'optuna' && !newOptimizer.optuna_config) {
        newOptimizer.optuna_config = { sampler: {} }; // Initialize sampler as well
      }

      return { ...prevConfig, optimizer: newOptimizer };
    });
  };

  const handleOptimizerNestedInputChange = (optimizerType, section, e) => {
    const { name, value, type, checked } = e.target;
    setConfig(prevConfig => ({
      ...prevConfig,
      optimizer: {
        ...prevConfig.optimizer,
        [optimizerType]: {
          ...prevConfig.optimizer[optimizerType],
          [section]: {
            ...prevConfig.optimizer[optimizerType]?.[section], // Use optional chaining
            [name]: type === 'checkbox' ? checked : value
          }
        }
      }
    }));
  };

  const handleOptimizerSimpleInputChange = (optimizerType, e) => {
    const { name, value, type, checked } = e.target;
    setConfig(prevConfig => ({
      ...prevConfig,
      optimizer: { ...prevConfig.optimizer, [optimizerType]: { ...prevConfig.optimizer[optimizerType], [name]: type === 'checkbox' ? checked : value } }
    }));
  };

  return (
    <div>
      <h2>Main Configuration</h2>
      {isLoading && <p>Loading configuration...</p>}
      {error && <p>Error loading configuration: {error.message}</p>}
      {config && !isLoading && !error && (
        <form>
          <div>
            <label htmlFor="run_name">Run Name:</label>
            <input
              type="text"
              id="run_name"
              name="run_name"
              value={config.run_name || ''}
              onChange={handleInputChange}
            />
          </div>
          <div>
            <label htmlFor="hydra.run.dir">Hydra Run Directory:</label>
            <input
              type="text"
              id="hydra.run.dir"
              name="hydra.run.dir"
              value={config.hydra?.run?.dir || ''}
              onChange={handleInputChange}
            />
          </div>
          <div>
            <label htmlFor="webui">WebUI:</label>
            <select
              id="webui"
              name="webui"
              value={config.webui || ''}
              onChange={handleInputChange}
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
                value={config.configs_dir || ''}
                onChange={handleInputChange}
              />
            </div>
            <div>
              <label htmlFor="conversion_dir">Conversion Directory:</label>
              <input
                type="text"
                id="conversion_dir"
                name="conversion_dir"
                value={config.conversion_dir || ''}
                onChange={handleInputChange}
              />
            </div>
            <div>
              <label htmlFor="wildcards_dir">Wildcards Directory:</label>
              <input
                type="text"
                id="wildcards_dir"
                name="wildcards_dir"
                value={config.wildcards_dir || ''}
                onChange={handleInputChange}
              />
            </div>
            <div>
              <label htmlFor="scorer_model_dir">Scorer Model Directory:</label>
              <input
                type="text"
                id="scorer_model_dir"
                name="scorer_model_dir"
                value={config.scorer_model_dir || ''}
                onChange={handleInputChange}
              />
            </div>
          </div>

          {/* Model Inputs Section */}
          <div>
            <h3>Model Inputs</h3>
            {config.model_paths.map((path, index) => (
              <div key={index} style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
                <input
                  type="text"
                  value={path}
                  onChange={(e) => handleModelPathChange(index, e.target.value)}
                  style={{ flexGrow: 1, marginRight: '10px' }}
                />
                <button
                  type="button"
                  onClick={() => handleSetBaseModel(index)}
                  style={{ marginRight: '5px', backgroundColor: config.base_model_index === index ? 'lightblue' : '', cursor: 'pointer' }}
                >
                  Base
                </button>
                <button
                  type="button"
                  onClick={() => handleSetFallbackModel(index)}
                  style={{ marginRight: '10px', backgroundColor: config.fallback_model_index === index ? 'lightblue' : '', cursor: 'pointer' }}
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
                value={config.merge_method || ''}
                onChange={handleMergeInputChange}
              >
                {/* Options will be populated dynamically later */}
                <option value="">Select a merge method</option>
              </select>
            </div>
            <div>
              <label htmlFor="device">Device:</label>
              <select
                id="device"
                name="device"
                value={config.device || ''}
                onChange={handleMergeInputChange}
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
                value={config.threads || ''}
                onChange={handleMergeInputChange}
              />
            </div>
            <div>
              <label htmlFor="merge_dtype">Merge Data Type:</label>
              <select
                id="merge_dtype"
                name="merge_dtype"
                value={config.merge_dtype || ''}
                onChange={handleMergeInputChange}
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
                value={config.save_dtype || ''}
                onChange={handleMergeInputChange}
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
                value={config.optimization_mode || ''}
                onChange={handleOptimizationModeChange}
              >
                <option value="merge">merge</option>
                <option value="recipe">recipe</option>
                <option value="layer_adjust">layer_adjust</option>
              </select>
            </div>

            {config.optimization_mode === 'recipe' && (
              <div>
                <h4>Recipe Optimization Settings</h4>
                <div>
                  <label htmlFor="recipe_optimization.recipe_path">Recipe Path:</label>
                  <input
                    type="text"
                    id="recipe_optimization.recipe_path"
                    value={config.recipe_optimization?.recipe_path || ''}
                    onChange={(e) => handleRecipeInputChange('recipe_path', e)}
                  />
                </div>
                <div>
                  <label htmlFor="recipe_optimization.target_nodes">Target Nodes:</label>
                  {/* TODO: Implement a more user-friendly selector with backend integration */}
                  <input
                    type="text"
                    id="recipe_optimization.target_nodes"
                    value={Array.isArray(config.recipe_optimization?.target_nodes) ? config.recipe_optimization.target_nodes.join(', ') : (config.recipe_optimization?.target_nodes || '')}
                    onChange={(e) => handleRecipeInputChange('target_nodes', e)}
                  />
                </div>
                <div>
                  <label htmlFor="recipe_optimization.target_params">Target Params:</label>
                  {/* TODO: Implement a more user-friendly selector with backend integration */}
                  <input
                    type="text"
                    id="recipe_optimization.target_params"
                    value={Array.isArray(config.recipe_optimization?.target_params) ? config.recipe_optimization.target_params.join(', ') : (config.recipe_optimization?.target_params || '')}
                    onChange={(e) => handleRecipeInputChange('target_params', e)}
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
              <input type="checkbox" id="save_merge_artifacts" name="save_merge_artifacts" checked={config.save_merge_artifacts || false} onChange={(e) => handleCheckboxChange('save_merge_artifacts', e.target.checked)} />
            </div>
            <div>
              <label htmlFor="save_best">Save Best:</label>
              <input type="checkbox" id="save_best" name="save_best" checked={config.save_best || false} onChange={(e) => handleCheckboxChange('save_best', e.target.checked)} />
            </div>
          </div>

          {/* Optimizer Configuration Section */}
          <div>
            <h3>Optimizer Configuration</h3>
            {/* Optimizer Selection (Dropdown) */}
            <div>
              <label>Select Optimizer:</label>
              <select
                value={config.optimizer?.bayes ? 'bayes' : (config.optimizer?.optuna ? 'optuna' : '')}
                onChange={(e) => handleOptimizerChange(e.target.value)}
              >
                <option value="">Select an optimizer</option>
                <option value="bayes">Bayes</option>
                <option value="optuna">Optuna</option>
              </select>
            </div>

            {config.optimizer?.bayes && (
              <div>
                <h4>Bayes Optimizer Settings</h4>
                <div>
                  <label htmlFor="bayes_config.load_log_file">Load Log File:</label>
                  <input
                    type="text"
                    id="bayes_config.load_log_file"
                    value={config.optimizer.bayes_config?.load_log_file ?? ''}
                    onChange={(e) => handleOptimizerNestedInputChange('bayes_config', 'load_log_file', e)}
                  />
                </div>
                <div>
                  <label htmlFor="bayes_config.reset_log_file">Reset Log File:</label>
                  <input
                    type="checkbox"
                    id="bayes_config.reset_log_file" // Use id for htmlFor
                    checked={config.optimizer.bayes_config?.reset_log_file || false}
                    onChange={(e) => handleOptimizerNestedInputChange('bayes_config', 'reset_log_file', e)}
                  />
                </div>
                <div>
                  <label htmlFor="bayes_config.sampler">Sampler:</label>
                  <input
                    type="text"
                    id="bayes_config.sampler"
                    value={config.optimizer.bayes_config?.sampler ?? ''}
                    onChange={(e) => handleOptimizerNestedInputChange('bayes_config', 'sampler', e)}
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
 value={config.optimizer.bayes_config?.acquisition_function?.kind ?? ''}
                      onChange={(e) => handleOptimizerNestedInputChange('bayes_config', 'acquisition_function', e)}
                    />
                  </div>
                  <div>
                    <label htmlFor="bayes_config.acquisition_function.kappa">Kappa:</label>
                    <input
                      type="number"
                      id="bayes_config.acquisition_function.kappa"
 value={config.optimizer.bayes_config?.acquisition_function?.kappa ?? ''}
                      onChange={(e) => handleOptimizerNestedInputChange('bayes_config', 'acquisition_function', e)}
                    />
                  </div>
                  <div>
                    <label htmlFor="bayes_config.acquisition_function.xi">Xi:</label>
                    <input
                      type="number"
                      id="bayes_config.acquisition_function.xi"
 value={config.optimizer.bayes_config?.acquisition_function?.xi ?? ''}
                      onChange={(e) => handleOptimizerNestedInputChange('bayes_config', 'acquisition_function', e)}
                    />
                  </div>
                  <div>
                    <label htmlFor="bayes_config.acquisition_function.kappa_decay">Kappa Decay:</label>
                    <input
                      type="number"
                      id="bayes_config.acquisition_function.kappa_decay"
 value={config.optimizer.bayes_config?.acquisition_function?.kappa_decay ?? ''}
                      onChange={(e) => handleOptimizerNestedInputChange('bayes_config', 'acquisition_function', e)}
                    />
                  </div>
                  <div>
                    <label htmlFor="bayes_config.acquisition_function.kappa_decay_delay">Kappa Decay Delay:</label>
                    <input
                      type="text" // Can be int or string
                      id="bayes_config.acquisition_function.kappa_decay_delay"
 value={config.optimizer.bayes_config?.acquisition_function?.kappa_decay_delay ?? ''}
                      onChange={(e) => handleOptimizerNestedInputChange('bayes_config', 'acquisition_function', e)}
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
                      id="bayes_config.bounds_transformer.enabled"
                      checked={config.optimizer.bayes_config?.bounds_transformer?.enabled ?? false}
                      onChange={(e) => handleOptimizerNestedInputChange('bayes_config', 'bounds_transformer', e)}
                    />
                  </div>
                  <div>
                    <label htmlFor="bayes_config.bounds_transformer.gamma_osc">Gamma OSC:</label>
                    <input
                      type="number"
                      id="bayes_config.bounds_transformer.gamma_osc"
                      value={config.optimizer.bayes_config?.bounds_transformer?.gamma_osc ?? ''}
                      onChange={(e) => handleOptimizerNestedInputChange('bayes_config', 'bounds_transformer', e)}
                    />
                  </div>
                  <div>
                    <label htmlFor="bayes_config.bounds_transformer.gamma_pan">Gamma PAN:</label>
                    <input
                      type="number"
                      id="bayes_config.bounds_transformer.gamma_pan"
                      value={config.optimizer.bayes_config?.bounds_transformer?.gamma_pan ?? ''}
                      onChange={(e) => handleOptimizerNestedInputChange('bayes_config', 'bounds_transformer', e)}
                    />
                  </div>
                  <div>
                    <label htmlFor="bayes_config.bounds_transformer.eta">Eta:</label>
                    <input
                      type="number"
                      id="bayes_config.bounds_transformer.eta"
                      value={config.optimizer.bayes_config?.bounds_transformer?.eta ?? ''}
                      onChange={(e) => handleOptimizerNestedInputChange('bayes_config', 'bounds_transformer', e)}
                    />
                  </div>
                  <div>
                    <label htmlFor="bayes_config.bounds_transformer.minimum_window">Minimum Window:</label>
                    <input
                      type="number"
                      id="bayes_config.bounds_transformer.minimum_window"
                      value={config.optimizer.bayes_config?.bounds_transformer?.minimum_window ?? ''}
                      onChange={(e) => handleOptimizerNestedInputChange('bayes_config', 'bounds_transformer', e)}
                    />
                  </div>
                </div>
              </div>
            )}

            {config.optimizer?.optuna && (
              <div>
                <h4>Optuna Optimizer Settings</h4>
                <div>
                  <label htmlFor="optuna_config.storage_dir">Storage Directory:</label>
                  <input
                    type="text"
                    id="optuna_config.storage_dir"
 value={config.optimizer.optuna_config?.storage_dir ?? ''}
                    onChange={(e) => handleOptimizerSimpleInputChange('optuna_config', e)}
                  />
                </div>
                <div>
                  <label htmlFor="optuna_config.resume_study_name">Resume Study Name:</label>
                  <input
                    type="text"
 id="optuna_config.resume_study_name"
                    value={config.optimizer.optuna_config?.resume_study_name || ''}
                    onChange={(e) => handleOptimizerSimpleInputChange('optuna_config', e)}
                  />
                </div>
                <div>
                  <label htmlFor="optuna_config.use_pruning">Use Pruning:</label>
                  <input
                    type="checkbox"
                    id="optuna_config.use_pruning"
 checked={config.optimizer.optuna_config?.use_pruning ?? false}
                    onChange={(e) => handleOptimizerSimpleInputChange('optuna_config', e)}
                  />
                </div>
                <div>
                  <label htmlFor="optuna_config.pruner_type">Pruner Type:</label>
                  <input
                    type="text"
 id="optuna_config.pruner_type"
                    value={config.optimizer.optuna_config?.pruner_type || ''}
                    onChange={(e) => handleOptimizerSimpleInputChange('optuna_config', e)}
                  />
                </div>
                <div>
                  <label htmlFor="optuna_config.early_stopping">Early Stopping:</label>
                  <input
                    type="checkbox"
                    id="optuna_config.early_stopping"
 checked={config.optimizer.optuna_config?.early_stopping ?? false}
                    onChange={(e) => handleOptimizerSimpleInputChange('optuna_config', e)}
                  />
                </div>
                <div>
                  <label htmlFor="optuna_config.patience">Patience:</label>
                  <input
                    type="number"
 id="optuna_config.patience"
                    value={config.optimizer.optuna_config?.patience || ''}
                    onChange={(e) => handleOptimizerSimpleInputChange('optuna_config', e)}
                  />
                </div>
                <div>
                  <label htmlFor="optuna_config.min_improvement">Minimum Improvement:</label>
                  <input
                    type="number"
 id="optuna_config.min_improvement"
                    value={config.optimizer.optuna_config?.min_improvement || ''}
                    onChange={(e) => handleOptimizerSimpleInputChange('optuna_config', e)}
                  />
                </div>
                <div>
                  <label htmlFor="optuna_config.n_jobs">N Jobs:</label>
                  <input
                    type="number"
 id="optuna_config.n_jobs"
                    value={config.optimizer.optuna_config?.n_jobs || ''}
                    onChange={(e) => handleOptimizerSimpleInputChange('optuna_config', e)}
                  />
                </div>
                <div>
                  <label htmlFor="optuna_config.launch_dashboard">Launch Dashboard:</label>
                  <input
                    type="checkbox"
                    id="optuna_config.launch_dashboard"
 checked={config.optimizer.optuna_config?.launch_dashboard ?? false}
                    onChange={(e) => handleOptimizerSimpleInputChange('optuna_config', e)}
                  />
                </div>
                <div>
                  <label htmlFor="optuna_config.dashboard_port">Dashboard Port:</label>
                  <input
                    type="number"
 id="optuna_config.dashboard_port"
                    value={config.optimizer.optuna_config?.dashboard_port || ''}
                    onChange={(e) => handleOptimizerSimpleInputChange('optuna_config', e)}
                  />
                </div>

                {/* Optuna Sampler Settings */}
                <div>
                  <h5>Optuna Sampler</h5>
                  <div>
                    <label htmlFor="optuna_config.sampler.type">Type:</label>
                    <input
                      type="text"
 id="optuna_config.sampler.type"
 value={config.optimizer.optuna_config?.sampler?.type ?? ''}
                      onChange={(e) => handleOptimizerNestedInputChange('optuna_config', 'sampler', e)}
                    />
                  </div>
                  <div>
                    <label htmlFor="optuna_config.sampler.multivariate">Multivariate:</label>
                    <input
                      type="checkbox"
 id="optuna_config.sampler.multivariate"
 checked={config.optimizer.optuna_config?.sampler?.multivariate ?? false} // Use checked for checkbox
                      onChange={(e) => handleOptimizerNestedInputChange('optuna_config', 'sampler', e)}
                    />
                  </div>
                  <div>
                    <label htmlFor="optuna_config.sampler.group">Group:</label>
                    <input
                      type="checkbox"
 id="optuna_config.sampler.group"
 checked={config.optimizer.optuna_config?.sampler?.group ?? false} // Use checked for checkbox
                      onChange={(e) => handleOptimizerNestedInputChange('optuna_config', 'sampler', e)}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
          {/* Add more form fields here later */}
        </form>
      )}
    </div>
  );
};

export default MainConfigTab;
