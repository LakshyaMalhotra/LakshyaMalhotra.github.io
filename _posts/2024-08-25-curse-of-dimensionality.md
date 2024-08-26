## The curse of dimensionality
The curse of dimensionality arises from the fact that at higher dimensions, the features space becomes increasingly sparse **for a fixed-size training set**. Because of this, many machine learning algorithms that work fine in low dimensional features space have their performance degraded severely in the features space in higher dimensions. Due to the sparsity, the entire training set only occupies a microscopic fraction of the features space. 

To quote Prof. Pedro Domingos of University of Washington in his excellent article: [A Few Useful Things to Know About Machine Learning](https://homes.cs.washington.edu/%7Epedrod/papers/cacm12.pdf):

>Even with a moderate features dimension of 100 and a huge training set of a trillion examples, the latter covers only a fraction of about $10^{-18}$ of the input space.

Let's try to understand this a bit more with a few simple mathematical intuitions:

**Intuition 1:** Consider a pizza with a rather odd slice which cuts it in two concentric circles. Let's also assume that the size of the pizza is significantly larger than the typical large size pizza.  For the sake of assigning some numbers, let's say that its radius is 1' whereas the the slice has a radius of 1/2'.

![intuition 1](pizza.png)

Let's calculate how much area is sliced out by the smaller circle:
$$A = \pi * r^{2}$$

$$ \implies A_{large} = \pi * 1 * 1 = \pi \quad sq.ft $$  

$$A_{small} = \pi * \frac{1}{2} * \frac{1}{2} = \frac{\pi}{4} \quad sq.ft$$

Remaining area = $A_{large} - A_{small} = \frac{3\pi}{4} \quad sq.ft$ . 

This means only 25% area is sliced out by the smaller pizza whereas 75% of the crust is still present in the remaining pizza.

Let's extrapolate this example to 3 dimensions to a sphere of a unit radius. If a smaller sphere of radius one-half is carved out. We can compute the volume remaining in the sphere in a way similar to the pizza example:

Remaining volume = $V_{large} - V_{small} = \frac{4}{3} * \pi * 1^{3} - \frac{4}{3} * \pi * (\frac{1}{2})^{3} = \frac{4}{3} * \pi * \frac{7}{8}$ 
We can see that only 12.5% volume of the sphere is carved out and 87.5% of the volume is still there.

This can be generalized to higher dimensions &ndash; as we increase the dimensionality, the volume of a hypersphere is concentrated more and more in its outer skin. If a fixed-size training set is uniformly distributed in an $n$-dimensional hypersphere, most of them will be closer to the edge of the hypersphere and far apart from each other.

**Intuition 2:** Consider six points lying on the circumference of a circle of unit radius such that they are equidistant from each other. By simple geometry, you can find that the angle between any two adjacent points is 60 degrees.

![equidistant points 1](equidistant_points_circle.png)

However, if those same 6 points located on the outer surface of a unit sphere, the angle between the adjacent point increases from 60 degrees to 90 degrees. 

![equidistant points 2](equidistant_points_sphere.png)

This can be extrapolated to higher dimensions and can be inducted that the angle between adjacent points will keep on increasing.

From Prof. Pedro Domingos &ndash; Higher dimensions break down our intuitions because they are very difficult to understand. Naively, one might think that gathering more features never hurts, since at worst they provide no new information about the class. But in fact their benefits may be outweighed by the curse of dimensionality.

> **A word of caution:** Although distance as a metric works well in the above intuition qualitatively, and is traditionally used in $k$ dimensions ($k \le 3$), it should be avoided in the feature spaces with higher values of $k$ ($k \gt 3$). Because of this, algorithms which use L1 or L2 norms like K-nearest neighbors become ill-defined in higher dimensions. The article by Aggarwal et al., "[On the Surprising Behavior of Distance Metrics in Higher Dimensional Space](https://bib.dbvis.de/uploadedFiles/155.pdf)" discusses this in great detail.

<!-- ## Problems with K-Nearest Neighbors in higher dimensions -->

## Distance is an ill-defined metric in higher dimensions

The distribution of distances from the origin in various dimensions follows specific patterns. Here’s a summary for a random point uniformly distributed inside a unit hypercube or hypersphere. For general $n$ dimensions, the distance \( r \) from the origin follows a chi distribution with \( n \) dimensions. The probability density function (**PDF**) is given by &ndash; 

$$f(r) = \frac{r^{n-1} e^{-r^2/2}}{2^{n/2-1} \Gamma(n/2)}$$ 

for $r \geq 0$, where $\Gamma$ is the gamma function.

The chi distribution generalizes to various dimensions, with the shape and scale influenced by the number of dimensions. In high dimensions, distances tend to cluster around a certain value, leading to a more concentrated distribution.

<div>
<script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
<script type="text/javascript">/**
* plotly.js v2.34.0
* Copyright 2012-2024, Plotly, Inc.
* All rights reserved.
* Licensed under the MIT license
*/
/*! For license information please see plotly.min.js.LICENSE.txt */
</script>        
</div>

## Struggles of K-Nearest Neighbors

Ratio plot:

<div id="e3b8cb53-ae68-4662-af5e-a34d2eccb8b7" data-root-id="p1004" style="display: contents;"></div>
<script type="application/json" id="d8e29e65-db80-43be-94d0-1fd9c5a629cd">
{"7906b2b6-1a8f-48e0-8cc2-c83f79e0403f":{"version":"3.5.2","title":"Bokeh Application","roots":[{"type":"object","name":"Figure","id":"p1004","attributes":{"height":400,"sizing_mode":"stretch_width","x_range":{"type":"object","name":"DataRange1d","id":"p1005"},"y_range":{"type":"object","name":"DataRange1d","id":"p1006"},"x_scale":{"type":"object","name":"LogScale","id":"p1014"},"y_scale":{"type":"object","name":"LogScale","id":"p1015"},"title":{"type":"object","name":"Title","id":"p1007","attributes":{"text":"Ratio of Max to Min Distance vs. Dimensions"}},"renderers":[{"type":"object","name":"GlyphRenderer","id":"p1045","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"p1001","attributes":{"selected":{"type":"object","name":"Selection","id":"p1002","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"p1003"},"data":{"type":"map","entries":[["dimensions",[2,10,50,100,200,500,1000,2000,5000]],["ratios",[337.10935795165784,24.975784934150024,2.9892181767564687,3.4097568395284656,3.19353853144922,3.078056158613766,3.1364957908890307,2.971440825684294,2.981772850079792]]]}}},"view":{"type":"object","name":"CDSView","id":"p1046","attributes":{"filter":{"type":"object","name":"AllIndices","id":"p1047"}}},"glyph":{"type":"object","name":"Scatter","id":"p1042","attributes":{"x":{"type":"field","field":"dimensions"},"y":{"type":"field","field":"ratios"},"size":{"type":"value","value":15},"line_color":{"type":"value","value":"navy"},"line_alpha":{"type":"value","value":0.5},"fill_color":{"type":"value","value":"navy"},"fill_alpha":{"type":"value","value":0.5},"hatch_color":{"type":"value","value":"navy"},"hatch_alpha":{"type":"value","value":0.5}}},"nonselection_glyph":{"type":"object","name":"Scatter","id":"p1043","attributes":{"x":{"type":"field","field":"dimensions"},"y":{"type":"field","field":"ratios"},"size":{"type":"value","value":15},"line_color":{"type":"value","value":"navy"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"navy"},"fill_alpha":{"type":"value","value":0.1},"hatch_color":{"type":"value","value":"navy"},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Scatter","id":"p1044","attributes":{"x":{"type":"field","field":"dimensions"},"y":{"type":"field","field":"ratios"},"size":{"type":"value","value":15},"line_color":{"type":"value","value":"navy"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"navy"},"fill_alpha":{"type":"value","value":0.2},"hatch_color":{"type":"value","value":"navy"},"hatch_alpha":{"type":"value","value":0.2}}}}}],"toolbar":{"type":"object","name":"Toolbar","id":"p1013","attributes":{"tools":[{"type":"object","name":"PanTool","id":"p1026"},{"type":"object","name":"WheelZoomTool","id":"p1027","attributes":{"renderers":"auto"}},{"type":"object","name":"BoxZoomTool","id":"p1028","attributes":{"overlay":{"type":"object","name":"BoxAnnotation","id":"p1029","attributes":{"syncable":false,"line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5,"level":"overlay","visible":false,"left":{"type":"number","value":"nan"},"right":{"type":"number","value":"nan"},"top":{"type":"number","value":"nan"},"bottom":{"type":"number","value":"nan"},"left_units":"canvas","right_units":"canvas","top_units":"canvas","bottom_units":"canvas","handles":{"type":"object","name":"BoxInteractionHandles","id":"p1035","attributes":{"all":{"type":"object","name":"AreaVisuals","id":"p1034","attributes":{"fill_color":"white","hover_fill_color":"lightgray"}}}}}}}},{"type":"object","name":"SaveTool","id":"p1036"},{"type":"object","name":"ResetTool","id":"p1037"},{"type":"object","name":"HelpTool","id":"p1038"},{"type":"object","name":"HoverTool","id":"p1050","attributes":{"renderers":"auto","tooltips":[["Dimensions","@dimensions"],["Ratio","@ratios"]]}}]}},"left":[{"type":"object","name":"LogAxis","id":"p1021","attributes":{"ticker":{"type":"object","name":"LogTicker","id":"p1022","attributes":{"num_minor_ticks":10,"mantissas":[1,5]}},"formatter":{"type":"object","name":"LogTickFormatter","id":"p1023"},"axis_label":"Max Distance / Min Distance","major_label_policy":{"type":"object","name":"AllLabels","id":"p1024"}}}],"below":[{"type":"object","name":"LogAxis","id":"p1016","attributes":{"ticker":{"type":"object","name":"LogTicker","id":"p1017","attributes":{"num_minor_ticks":10,"mantissas":[1,5]}},"formatter":{"type":"object","name":"LogTickFormatter","id":"p1018"},"axis_label":"Number of Dimensions","major_label_policy":{"type":"object","name":"AllLabels","id":"p1019"}}}],"center":[{"type":"object","name":"Grid","id":"p1020","attributes":{"axis":{"id":"p1016"}}},{"type":"object","name":"Grid","id":"p1025","attributes":{"dimension":1,"axis":{"id":"p1021"}}},{"type":"object","name":"Legend","id":"p1048","attributes":{"items":[{"type":"object","name":"LegendItem","id":"p1049","attributes":{"label":{"type":"value","value":"Data Points"},"renderers":[{"id":"p1045"}]}}]}}]}}]}}
</script>
<script type="text/javascript">
    (function() {
    const fn = function() {
        Bokeh.safely(function() {
        (function(root) {
            function embed_document(root) {
            const docs_json = document.getElementById('d8e29e65-db80-43be-94d0-1fd9c5a629cd').textContent;
            const render_items = [{"docid":"7906b2b6-1a8f-48e0-8cc2-c83f79e0403f","roots":{"p1004":"e3b8cb53-ae68-4662-af5e-a34d2eccb8b7"},"root_ids":["p1004"]}];
            root.Bokeh.embed.embed_items(docs_json, render_items);
            }
            if (root.Bokeh !== undefined) {
            embed_document(root);
            } else {
            let attempts = 0;
            const timer = setInterval(function(root) {
                if (root.Bokeh !== undefined) {
                clearInterval(timer);
                embed_document(root);
                } else {
                attempts++;
                if (attempts > 100) {
                    clearInterval(timer);
                    console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                }
                }
            }, 10, root)
            }
        })(window);
        });
    };
    if (document.readyState != "loading") fn();
    else document.addEventListener("DOMContentLoaded", fn);
    })();
</script>

## Principal Component Analysis
It's a tool for providing a low-dimensional representation of the dataset without losing much of the information. By information we meant, it tries to capture as much variation in the data as possible.

The new features can be projected in a much smaller space whose axes are called principal components. These axes are orthogonal to each other and their direction can be interpreted as the direction of maximum variance. The orthogonality between the principal components ensures no correlation between each other. 

PCA is very sensitive to the data scaling. Features with larger scales or variances will completely dominate the PCA results. PCA directions will be biased towards these features, which will potentially lead to the incorrect representations of the structure of the data. It is therefore highly recommended to standardize the data first before applying it. 

The overall flow for applying the PCA for dimensionality reduction can be summarized by following steps:
1. Normalize the dataset containing $d$ features with zero mean and unit standard deviation 
2. Construct the covariance matrix
3. Compute the eigenvectors and corresponding eigenvalues of this matrix
4. Sort the eigenvalues in the decreasing order
5. Select $k$ ($\lt d$) eigenvectors that correspond to the largest eigenvalues, these eigenvectors will form the basis of our new feature space. In other words, the set of these $k$ eigenvectors is the $k$ principal components of our dataset
6. Construct a new matrix from these $k$ eigenvectors
7. The last thing to do it to project our original dataset in the $d$ dimensional feature space to this new feature space with the $k$ features. This will be done by a simple matrix multiplication of the original dataset with the matrix we constructed in the last step
> NOTE: Refer to the notebook for more details


## References
- https://stats.stackexchange.com/questions/451027/mathematical-demonstration-of-the-distance-concentration-in-high-dimensions
- https://homes.cs.washington.edu/%7Epedrod/papers/cacm12.pdf
- https://mathoverflow.net/questions/128786/history-of-the-high-dimensional-volume-paradox/128881#128881
- https://stats.stackexchange.com/questions/99171/why-is-euclidean-distance-not-a-good-metric-in-high-dimensions
