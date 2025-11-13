// Global variables
let currentData = {};
let currentTarget = 'any';
let currentColorBy = 'auc';
let worldMap = null;
let tooltip = null;
let colorScale = null;

// Feature name mapping for better display
const featureNameDict = {
    "op_drain_yn": "Abdominal drain",
    "comorb_n": "Comorbidities (N)",
    "pre_urgency_Emergency": "Urgency of Surgery",
    "op_nassar": "Nassar Grade",
    "hist_ac": "Hx AC / Cholangitis",
    "op_approach_conv_reason_nan": "Reason for Open Conversion",
    "asa": "ASA Grade",
    "has_bleeding": "Has Bleeding",
    "op_img_stone_mx_nan": "Intraoperative CBD assessment findings",
    "op_approach_open_reason_Disease severity": "Reason for Open Approach",
    "op_contam": "Operative Contamination",
    "pre_img_ercp": "Pre-op ERCP",
    "pre_urgency_Delayed": "Urgency of Surgery",
    "op_approach_Open conversion": "Operative Approach",
    "op_img_stone_mx_No intraoperative treatment attempted": "Intraoperative CBD assessment findings",
    "has_bile_spilt": "Bile Spillage",
    "continent_Asia": "Continent",
    "op_approach_conv_reason_Suspected bile duct injury": "Reason for Open Conversion",
    "train_perform": "Trainees Perform Surgery",
    "service_chole_n": "Cholecystectomies / Year",
    "service_fu": "Post-op Follow-up",
    "hosp_mis_image_Yes (Photo and Video)": "Routine Intraop Images",
    "train_sim_yn": "Local Sim Training",
    "has_box_trainer": "Has Box Trainer",
    "specialty_colorectal": "Specialty",
    "diag_eus_yn": "EUS",
    "has_it_simulation_model": "Has IT Sim Model",
    "hosp_hpb_oncall": "HPB On-Call Services",
    "has_post_training_fellow": "Has Post-Training Fellow",
    "has_junior_trainee": "Has Junior Trainee",
    "wb": "Country Income",
    "service_cons_n": "Number of Consultants",
    "hosp_mis_image_Yes (Photo only)": "Routine Intraop Images",
    "diag_hist": "Histological Exam after Surgery",
    "diag_ostomy_oncall": "On-Call Cholecystostomy",
    "has_senior_trainee": "Has Senior Trainee",
    "diag_mrcp_yn": "MRCP",
    "specialty_general": "Specialty",
    "diag_ioc": "Intraop IOC",
    "frailty": "Frailty",
    "op_operator_cons_Vascular": "Operator Specialty",
    "pre_indication_tg": "Pre-op Indication",
    "op_cbd_explore_No": "CBD Exploration",
    "has_stones_spilt": "Stones Spillage",
    "pt_diabetes": "Patient Diabetes",
    "op_approach_mis_reuse": "Operative Approach",
    "pre_img_ct": "Pre-op CT",
    "op_performed_stc_nan": "Performed STC"
};

// Country code mapping from ISO to numeric IDs used in the world map
const countryCodeMapping = {
    'AF': '004', 'AL': '008', 'DZ': '012', 'AS': '016', 'AD': '020', 'AO': '024', 'AI': '660', 'AQ': '010', 'AG': '028', 'AR': '032',
    'AM': '051', 'AW': '533', 'AU': '036', 'AT': '040', 'AZ': '031', 'BS': '044', 'BH': '048', 'BD': '050', 'BB': '052', 'BY': '112',
    'BE': '056', 'BZ': '084', 'BJ': '204', 'BM': '060', 'BT': '064', 'BO': '068', 'BA': '070', 'BW': '072', 'BV': '074', 'BR': '076',
    'IO': '086', 'BN': '096', 'BG': '100', 'BF': '854', 'BI': '108', 'KH': '116', 'CM': '120', 'CA': '124', 'CV': '132', 'KY': '136',
    'CF': '140', 'TD': '148', 'CL': '152', 'CN': '156', 'CX': '162', 'CC': '166', 'CO': '170', 'KM': '174', 'CG': '178', 'CD': '180',
    'CK': '184', 'CR': '188', 'CI': '384', 'HR': '191', 'CU': '192', 'CY': '196', 'CZ': '203', 'DK': '208', 'DJ': '262', 'DM': '212',
    'DO': '214', 'EC': '218', 'EG': '818', 'SV': '222', 'GQ': '226', 'ER': '232', 'EE': '233', 'ET': '231', 'FK': '238', 'FO': '234',
    'FJ': '242', 'FI': '246', 'FR': '250', 'GF': '254', 'PF': '258', 'TF': '260', 'GA': '266', 'GM': '270', 'GE': '268', 'DE': '276',
    'GH': '288', 'GI': '292', 'GR': '300', 'GL': '304', 'GD': '308', 'GP': '312', 'GU': '316', 'GT': '320', 'GN': '324', 'GW': '624',
    'GY': '328', 'HT': '332', 'HM': '334', 'VA': '336', 'HN': '340', 'HK': '344', 'HU': '348', 'IS': '352', 'IN': '356', 'ID': '360',
    'IR': '364', 'IQ': '368', 'IE': '372', 'IL': '376', 'IT': '380', 'JM': '388', 'JP': '392', 'JO': '400', 'KZ': '398', 'KE': '404',
    'KI': '296', 'KP': '408', 'KR': '410', 'KW': '414', 'KG': '417', 'LA': '418', 'LV': '428', 'LB': '422', 'LS': '426', 'LR': '430',
    'LY': '434', 'LI': '438', 'LT': '440', 'LU': '442', 'MO': '446', 'MK': '807', 'MG': '450', 'MW': '454', 'MY': '458', 'MV': '462',
    'ML': '466', 'MT': '470', 'MH': '584', 'MQ': '474', 'MR': '478', 'MU': '480', 'YT': '175', 'MX': '484', 'FM': '583', 'MD': '498',
    'MC': '492', 'MN': '496', 'MS': '500', 'MA': '504', 'MZ': '508', 'MM': '104', 'NA': '516', 'NR': '520', 'NP': '524', 'NL': '528',
    'NC': '540', 'NZ': '554', 'NI': '558', 'NE': '562', 'NG': '566', 'NU': '570', 'NF': '574', 'MP': '580', 'NO': '578', 'OM': '512',
    'PK': '586', 'PW': '585', 'PS': '275', 'PA': '591', 'PG': '598', 'PY': '600', 'PE': '604', 'PH': '608', 'PN': '612', 'PL': '616',
    'PT': '620', 'PR': '630', 'QA': '634', 'RE': '638', 'RO': '642', 'RU': '643', 'RW': '646', 'SH': '654', 'KN': '659', 'LC': '662',
    'PM': '666', 'VC': '670', 'WS': '882', 'SM': '674', 'ST': '678', 'SA': '682', 'SN': '686', 'SC': '690', 'SL': '694', 'SG': '702',
    'SK': '703', 'SI': '705', 'SB': '090', 'SO': '706', 'ZA': '710', 'GS': '239', 'ES': '724', 'LK': '144', 'SD': '729', 'SR': '740',
    'SJ': '744', 'SZ': '748', 'SE': '752', 'CH': '756', 'SY': '760', 'TJ': '762', 'TZ': '834', 'TH': '764', 'TL': '626',
    'TG': '768', 'TK': '772', 'TO': '776', 'TT': '780', 'TN': '788', 'TR': '792', 'TM': '795', 'TC': '796', 'TV': '798', 'UG': '800',
    'UA': '804', 'AE': '784', 'GB': '826', 'US': '840', 'UM': '581', 'UY': '858', 'UZ': '860', 'VU': '548', 'VE': '862', 'VN': '704',
    'VG': '092', 'VI': '850', 'WF': '876', 'EH': '732', 'YE': '887', 'ZM': '894', 'ZW': '716'
};

// Target configurations
const targets = {
    any: {
        name: 'Any Complications',
        file: 'any.csv'
    },
    chole: {
        name: 'Cholecystectomy Complications',
        file: 'chole.csv'
    },
    major: {
        name: 'Major Complications',
        file: 'major.csv'
    }
};

// Color schemes for different metrics
const colorSchemes = {
    auc: ['#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15'],
    ap: ['#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15'],
    n_patients: ['#edf8e9', '#bae4b3', '#74c476', '#31a354', '#006d2c']
};

// Initialize the application
async function init() {
    await loadAllData();
    setupEventListeners();
    createTooltip();
    updateMap();
}

// Load all data files
async function loadAllData() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading';
    loadingDiv.textContent = 'Loading data...';
    document.getElementById('map').appendChild(loadingDiv);

    try {
        for (const [target, config] of Object.entries(targets)) {
            console.log(`Loading ${config.file} for ${target}...`);
            const response = await fetch(config.file);
            
            if (!response.ok) {
                throw new Error(`Failed to load ${config.file}: ${response.status} ${response.statusText}`);
            }
            
            const csvText = await response.text();
            console.log(`Successfully loaded ${config.file}, ${csvText.length} characters`);
            currentData[target] = parseCSV(csvText);
            console.log(`Parsed ${target} data:`, Object.keys(currentData[target]).length, 'countries');
        }
        document.getElementById('map').innerHTML = '';
        console.log('All data loaded successfully');
    } catch (error) {
        console.error('Error loading data:', error);
        
        // Show more detailed error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'no-data';
        errorDiv.innerHTML = `
            <h3>Error Loading Data</h3>
            <p>${error.message}</p>
            <p><strong>Possible solutions:</strong></p>
            <ul style="text-align: left; max-width: 500px; margin: 20px auto;">
                <li>Make sure all CSV files are in the same folder as index.html</li>
                <li>Try running a local server instead of opening the file directly</li>
                <li>Check browser console for more details</li>
            </ul>
            <p><strong>To run a local server:</strong></p>
            <code style="background: #f0f0f0; padding: 5px; border-radius: 3px;">
                python -m http.server 8000
            </code>
            <p>Then open <a href="http://localhost:8000" target="_blank">http://localhost:8000</a></p>
        `;
        document.getElementById('map').innerHTML = '';
        document.getElementById('map').appendChild(errorDiv);
    }
}

// Parse CSV data
function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    const headers = lines[0].split(',');
    const data = {};

    console.log('CSV headers:', headers);

    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',');
        const country = values[5]; // country_iso column
        const row = {};

        headers.forEach((header, index) => {
            let value = values[index];
            if (value === '') {
                value = null;
            } else if (!isNaN(value)) {
                value = parseFloat(value);
            }
            row[header.trim()] = value;
        });

        data[country] = row;
    }

    return data;
}

// Setup event listeners
function setupEventListeners() {
    // Tab switching
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            currentTarget = tab.dataset.target;
            updateMap();
        });
    });

    // Color by selection
    document.getElementById('colorBy').addEventListener('change', (e) => {
        currentColorBy = e.target.value;
        updateMap();
    });

    // GitHub button
    document.getElementById('githubBtn').addEventListener('click', (e) => {
        // The link is already set in HTML, this is just for any additional functionality
        console.log('GitHub repository opened');
    });

    // Download button
    document.getElementById('downloadBtn').addEventListener('click', () => {
        downloadCurrentData();
    });
}

// Create tooltip element
function createTooltip() {
    tooltip = d3.select('body')
        .append('div')
        .attr('class', 'tooltip')
        .style('opacity', 0);
}

// Update the map
async function updateMap() {
    const data = currentData[currentTarget];
    if (!data) {
        console.error('No data available for target:', currentTarget);
        return;
    }

    // Update title
    const colorByLabels = {
        auc: 'AUROC',
        ap: 'AUPRC',
        n_patients: 'Number of Patients'
    };
    document.getElementById('mapTitle').textContent =
        `${targets[currentTarget].name} - ${colorByLabels[currentColorBy]}`;

    // Create color scale
    const values = Object.values(data)
        .map(d => d[currentColorBy])
        .filter(v => v !== null && !isNaN(v));
    
    if (values.length === 0) {
        console.error('No valid values found for', currentColorBy);
        return;
    }
    
    colorScale = d3.scaleQuantile()
        .domain(values)
        .range(colorSchemes[currentColorBy]);

    // Load and render world map
    await loadWorldMap(data);
    updateLegend();
}

// Load and render world map
async function loadWorldMap(data) {
    const mapContainer = document.getElementById('map');
    mapContainer.innerHTML = '';

    const width = mapContainer.clientWidth;
    const height = 600;

    const svg = d3.select('#map')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    // Use a better projection that fills the container
    const projection = d3.geoEqualEarth()
        .fitSize([width, height], { type: 'Sphere' });
        // const projection = d3.geoMercator()

    const path = d3.geoPath().projection(projection);

    try {
        // Load world map data from local file
        console.log('Loading world map data from local file...');
        const world = await d3.json('countries-110m.json');
        const countries = topojson.feature(world, world.objects.countries);
        console.log('World map loaded, countries:', countries.features.length);

        // Create country paths
        svg.selectAll('path')
            .data(countries.features)
            .enter()
            .append('path')
            .attr('d', path)
            .attr('fill', d => {
                // Convert numeric ID to ISO code for data lookup
                const isoCode = Object.keys(countryCodeMapping).find(key => countryCodeMapping[key] === d.id);
                const countryData = data[isoCode];
                if (countryData && countryData[currentColorBy] !== null) {
                    return colorScale(countryData[currentColorBy]);
                }
                return '#ddd';
            })
            .attr('stroke', '#fff')
            .attr('stroke-width', 0.5)
            .style('cursor', 'pointer')
            .on('mouseover', function(event, d) {
                const isoCode = Object.keys(countryCodeMapping).find(key => countryCodeMapping[key] === d.id);
                const countryData = data[isoCode];
                if (countryData) {
                    showTooltip(event, d, countryData);
                }
            })
            .on('mouseout', hideTooltip)
            .on('click', function(event, d) {
                const isoCode = Object.keys(countryCodeMapping).find(key => countryCodeMapping[key] === d.id);
                const countryData = data[isoCode];
                if (countryData) {
                    showDetailedTooltip(event, d, countryData);
                }
            });

        console.log('Map rendered successfully');

    } catch (error) {
        console.error('Error loading world map:', error);
        mapContainer.innerHTML = `
            <div class="no-data">
                <h3>Error Loading World Map</h3>
                <p>Unable to load world map data from local file.</p>
                <p><strong>Error:</strong> ${error.message}</p>
                <p><strong>Possible solutions:</strong></p>
                <ul style="text-align: left; max-width: 500px; margin: 20px auto;">
                    <li>Make sure 'countries-110m.json' is in the same folder as index.html</li>
                    <li>Check that the JSON file is not corrupted</li>
                    <li>Try refreshing the page</li>
                </ul>
            </div>
        `;
    }
}

// Show tooltip on hover
function showTooltip(event, d, countryData) {
    const colorByLabels = {
        auc: 'AUROC',
        ap: 'AUPRC',
        n_patients: 'Number of Patients'
    };

    const value = countryData[currentColorBy];
    const displayValue = value !== null ?
        (currentColorBy === 'n_patients' ? value.toLocaleString() :
         value.toFixed(3)) : 'N/A';

    // Convert numeric ID to ISO code for country name lookup
    const isoCode = Object.keys(countryCodeMapping).find(key => countryCodeMapping[key] === d.id);

    // Create feature importance data for top 5 features
    const features = [];
    for (let i = 1; i <= 5; i++) {
        const name = countryData[`top_feature_${i}_name`];
        const importance = countryData[`top_feature_${i}_importance`];
        if (name && importance !== null) {
            features.push({
                name: featureNameDict[name] || name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
                importance: importance
            });
        }
    }

    // Create bar chart HTML for top 5 features
    const chartHtml = features.length > 0 ? `
        <div style="margin-top: 15px;">
            <h4 style="margin-bottom: 10px; color: #ffd700;">Top 5 Features</h4>
            <div style="max-height: 150px; overflow-y: auto;">
                ${features.map(f => `
                    <div style="margin: 8px 0; display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 0.75rem; flex: 1; margin-right: 10px;">${f.name}</span>
                        <div style="width: 80px; height: 12px; background: #333; border-radius: 6px; margin-left: 10px;">
                            <div style="width: ${f.importance * 100}%; height: 100%; background: #ffd700; border-radius: 6px;"></div>
                        </div>
                        <span style="font-size: 0.75rem; margin-left: 10px; min-width: 35px;">${(f.importance * 100).toFixed(1)}%</span>
                    </div>
                `).join('')}
            </div>
        </div>
    ` : '';

    tooltip.transition()
        .duration(200)
        .style('opacity', 0.9);

    tooltip.html(`
        <h3>${getCountryName(isoCode)}</h3>
        <div class="metric">
            <span>${colorByLabels[currentColorBy]}:</span>
            <span>${displayValue}</span>
        </div>
        <div class="metric">
            <span>Patients:</span>
            <span>${countryData.n_patients?.toLocaleString() || 'N/A'}</span>
        </div>
        ${currentColorBy !== 'auc' ? `<div class="metric">
            <span>AUROC:</span>
            <span>${countryData.auc ? countryData.auc.toFixed(3) : 'N/A'}</span>
        </div>` : ''}
        ${currentColorBy !== 'ap' ? `<div class="metric">
            <span>AUPRC:</span>
            <span>${countryData.ap ? countryData.ap.toFixed(3) : 'N/A'}</span>
        </div>` : ''}
        ${chartHtml}
    `)
    .style('left', (event.pageX + 10) + 'px')
    .style('top', (event.pageY - 28) + 'px');
}

// Show detailed tooltip with feature importance chart
function showDetailedTooltip(event, d, countryData) {
    const colorByLabels = {
        auc: 'AUROC',
        ap: 'AUPRC',
        n_patients: 'Number of Patients'
    };

    const value = countryData[currentColorBy];
    const displayValue = value !== null ?
        (currentColorBy === 'n_patients' ? value.toLocaleString() :
         value.toFixed(3)) : 'N/A';

    // Convert numeric ID to ISO code for country name lookup
    const isoCode = Object.keys(countryCodeMapping).find(key => countryCodeMapping[key] === d.id);

    // Create feature importance data for top 5 features
    const features = [];
    for (let i = 1; i <= 5; i++) {
        const name = countryData[`top_feature_${i}_name`];
        const importance = countryData[`top_feature_${i}_importance`];
        if (name && importance !== null) {
            features.push({
                name: featureNameDict[name] || name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
                importance: importance
            });
        }
    }

    // Create chart HTML for top 5 features
    const chartHtml = features.length > 0 ? `
        <div style="margin-top: 15px;">
            <h4 style="margin-bottom: 10px; color: #ffd700;">Top 5 Features</h4>
            <div style="max-height: 150px; overflow-y: auto;">
                ${features.map(f => `
                    <div style="margin: 8px 0; display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 0.75rem; flex: 1; margin-right: 10px;">${f.name}</span>
                        <div style="width: 80px; height: 12px; background: #333; border-radius: 6px; margin-left: 10px;">
                            <div style="width: ${f.importance * 100}%; height: 100%; background: #ffd700; border-radius: 6px;"></div>
                        </div>
                        <span style="font-size: 0.75rem; margin-left: 10px; min-width: 35px;">${(f.importance * 100).toFixed(1)}%</span>
                    </div>
                `).join('')}
            </div>
        </div>
    ` : '';

    tooltip.transition()
        .duration(200)
        .style('opacity', 0.9);

    tooltip.html(`
        <h3>${getCountryName(isoCode)}</h3>
        <div class="metric">
            <span>${colorByLabels[currentColorBy]}:</span>
            <span>${displayValue}</span>
        </div>
        <div class="metric">
            <span>Patients:</span>
            <span>${countryData.n_patients?.toLocaleString() || 'N/A'}</span>
        </div>
        ${currentColorBy !== 'auc' ? `<div class="metric">
            <span>AUROC:</span>
            <span>${countryData.auc ? countryData.auc.toFixed(3) : 'N/A'}</span>
        </div>` : ''}
        ${currentColorBy !== 'ap' ? `<div class="metric">
            <span>AUPRC:</span>
            <span>${countryData.ap ? countryData.ap.toFixed(3) : 'N/A'}</span>
        </div>` : ''}
        ${chartHtml}
    `)
    .style('left', (event.pageX + 10) + 'px')
    .style('top', (event.pageY - 28) + 'px');
}

// Hide tooltip
function hideTooltip() {
    tooltip.transition()
        .duration(500)
        .style('opacity', 0);
}

// Update legend
function updateLegend() {
    const legend = document.getElementById('legend');
    legend.innerHTML = '';

    const colorByLabels = {
        auc: 'AUROC',
        ap: 'AUPRC',
        n_patients: 'Number of Patients'
    };

    const data = currentData[currentTarget];
    if (!data) return;

    const values = Object.values(data)
        .map(d => d[currentColorBy])
        .filter(v => v !== null && !isNaN(v));

    if (values.length === 0) return;

    const min = Math.min(...values);
    const max = Math.max(...values);

    // Create color bar legend
    const legendContainer = document.createElement('div');
    legendContainer.style.cssText = `
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 10px;
        margin: 20px 0;
    `;

    // Create color bar
    const colorBar = document.createElement('div');
    colorBar.style.cssText = `
        width: 200px;
        height: 20px;
        background: linear-gradient(to right, ${colorSchemes[currentColorBy].join(', ')});
        border-radius: 4px;
        border: 1px solid #ccc;
    `;

    // Create labels container
    const labelsContainer = document.createElement('div');
    labelsContainer.style.cssText = `
        display: flex;
        justify-content: space-between;
        width: 200px;
        font-size: 12px;
        color: #666;
    `;

    // Create min and max labels
    const minLabel = document.createElement('span');
    const maxLabel = document.createElement('span');

    const minDisplayValue = currentColorBy === 'n_patients' ?
        min.toLocaleString() :
        min.toFixed(3);

    const maxDisplayValue = currentColorBy === 'n_patients' ?
        max.toLocaleString() :
        max.toFixed(3);

    minLabel.textContent = minDisplayValue;
    maxLabel.textContent = maxDisplayValue;

    labelsContainer.appendChild(minLabel);
    labelsContainer.appendChild(maxLabel);

    // Create title
    const title = document.createElement('div');
    title.textContent = colorByLabels[currentColorBy];
    title.style.cssText = `
        font-weight: bold;
        font-size: 14px;
        color: #333;
        margin-bottom: 5px;
    `;

    // Add "No Data" legend item
    const noDataItem = document.createElement('div');
    noDataItem.style.cssText = `
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 10px;
        font-size: 12px;
        color: #666;
    `;
    noDataItem.innerHTML = `
        <div style="width: 20px; height: 20px; background: #ddd; border-radius: 4px; border: 1px solid #ccc;"></div>
        <span>No Data</span>
    `;

    // Assemble legend
    legendContainer.appendChild(title);
    legendContainer.appendChild(colorBar);
    legendContainer.appendChild(labelsContainer);
    legendContainer.appendChild(noDataItem);

    legend.appendChild(legendContainer);
}

// Get country name from ISO code
function getCountryName(isoCode) {
    const countryNames = {
        'GB': 'United Kingdom', 'IN': 'India', 'IT': 'Italy', 'US': 'United States',
        'AU': 'Australia', 'MX': 'Mexico', 'ES': 'Spain', 'PK': 'Pakistan',
        'TR': 'Turkey', 'PT': 'Portugal', 'CO': 'Colombia', 'EG': 'Egypt',
        'GR': 'Greece', 'PH': 'Philippines', 'PE': 'Peru', 'SA': 'Saudi Arabia',
        'TN': 'Tunisia', 'LY': 'Libya', 'SE': 'Sweden', 'JO': 'Jordan',
        'NZ': 'New Zealand', 'DE': 'Germany', 'MY': 'Malaysia', 'PL': 'Poland',
        'HR': 'Croatia', 'AR': 'Argentina', 'BA': 'Bosnia and Herzegovina',
        'CN': 'China', 'YE': 'Yemen', 'RO': 'Romania', 'CZ': 'Czech Republic',
        'BR': 'Brazil', 'RU': 'Russia', 'HU': 'Hungary', 'MA': 'Morocco',
        'ET': 'Ethiopia', 'CH': 'Switzerland', 'TW': 'Taiwan', 'UA': 'Ukraine',
        'GT': 'Guatemala', 'CA': 'Canada', 'KW': 'Kuwait', 'RS': 'Serbia',
        'LT': 'Lithuania', 'FR': 'France', 'SD': 'Sudan', 'DO': 'Dominican Republic',
        'IE': 'Ireland', 'LV': 'Latvia', 'LK': 'Sri Lanka', 'BG': 'Bulgaria',
        'AZ': 'Azerbaijan', 'AE': 'United Arab Emirates', 'OM': 'Oman',
        'ZA': 'South Africa', 'KZ': 'Kazakhstan', 'AT': 'Austria', 'EC': 'Ecuador',
        'SO': 'Somalia', 'NG': 'Nigeria', 'BE': 'Belgium', 'QA': 'Qatar',
        'NL': 'Netherlands', 'CY': 'Cyprus', 'IQ': 'Iraq', 'ID': 'Indonesia',
        'NO': 'Norway', 'LB': 'Lebanon', 'PY': 'Paraguay', 'DZ': 'Algeria',
        'PS': 'Palestine', 'SY': 'Syria', 'TH': 'Thailand', 'KE': 'Kenya',
        'GH': 'Ghana', 'MT': 'Malta', 'UY': 'Uruguay', 'JP': 'Japan',
        'HK': 'Hong Kong', 'MK': 'North Macedonia', 'BY': 'Belarus',
        'BD': 'Bangladesh', 'SI': 'Slovenia', 'RW': 'Rwanda', 'IS': 'Iceland',
        'SG': 'Singapore', 'TZ': 'Tanzania', 'MN': 'Mongolia', 'VE': 'Venezuela',
        'GE': 'Georgia', 'SN': 'Senegal', 'SV': 'El Salvador', 'WD': 'Dominica',
        'MO': 'Macau', 'LU': 'Luxembourg', 'NA': 'Namibia', 'PA': 'Panama',
        'AW': 'Aruba', 'JM': 'Jamaica', 'UG': 'Uganda', 'BI': 'Burundi',
        'BF': 'Burkina Faso', 'VN': 'Vietnam', 'BJ': 'Benin', 'AL': 'Albania',
        'IR': 'Iran', 'CM': 'Cameroon', 'IL': 'Israel', 'GA': 'Gabon', 'KH': 'Cambodia'
    };
    
    return countryNames[isoCode] || isoCode;
}

// Download current data as CSV
function downloadCurrentData() {
    const data = currentData[currentTarget];
    if (!data) {
        alert('No data available for download');
        return;
    }

    // Get the original CSV filename for the current target
    const filenameMap = {
        'any': 'any.csv',
        'chole': 'chole.csv',
        'major': 'major.csv'
    };

    const filename = filenameMap[currentTarget];
    if (!filename) {
        alert('Invalid target for download');
        return;
    }

    // Create a download link
    const link = document.createElement('a');
    link.href = filename;
    link.download = filename;
    link.style.display = 'none';
    
    // Add to DOM, click, and remove
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    console.log(`Downloading ${filename}`);
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', init); 