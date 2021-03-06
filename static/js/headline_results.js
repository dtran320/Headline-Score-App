var needle;
var suggestions;

(function(){

var barWidth, chart_cg, chartInset, degToRad, repaintGauge,
    height, padRad, percToDeg, percToRad, chart_sl,
    percent, radius, sectionIndx, svg_cg, cg, totalPercent, width, svg_sl, sl;

  
  sectionPerc = 1 / 2;
  padRad = 0.025;
  chartInset = 10;
  totalPercent = .75;  // Orientation of gauge


  cg = d3.select('.chart-gauge');
  width = cg[0][0].offsetWidth; 
  height = 300;  
  radius = 150;  //radius of the gauge
  barWidth = 40; //width of the gauge bar
  
  /*
    Utility methods 
  */
  percToDeg = function(perc) {
    return perc * 360;
  };

  percToRad = function(perc) {
    return degToRad(percToDeg(perc));
  };

  degToRad = function(deg) {
    return deg * Math.PI / 180;
  };

  // Create SVG element
  svg_cg = cg.append('svg').attr('width', width).attr('height', height);

  // Add layer for the panel
  chart_cg = svg_cg.append('g').attr('transform', "translate(" + ((width) / 2) + ", " + ((height) / 2) + ")");
  chart_cg.append('path').attr('class', "arc chart-filled");
  chart_cg.append('path').attr('class', "arc chart-empty");

  arc2 = d3.svg.arc().outerRadius(radius - chartInset).innerRadius(radius - chartInset - barWidth)
  arc1 = d3.svg.arc().outerRadius(radius - chartInset).innerRadius(radius - chartInset - barWidth)
  
  
    
  //Add text to display the predicted score
  scoreText = chart_cg.append('text')
    .attr("x", -120) 
    .attr("y", 55)
    .style("font-size", "40px");

  //Repaint the gauge layer
  repaintGauge = function (perc) 
  {
    var next_start = totalPercent;
    arcStartRad = percToRad(next_start);
    arcEndRad = arcStartRad + percToRad(perc / 2);
    next_start += perc / 2;

    arc1.startAngle(arcStartRad).endAngle(arcEndRad);

    arcStartRad = percToRad(next_start);
    arcEndRad = arcStartRad + percToRad((1 - perc) / 2);

    arc2.startAngle(arcStartRad + padRad).endAngle(arcEndRad);


    chart_cg.select(".chart-filled").attr('d', arc1);
    chart_cg.select(".chart-empty").attr('d', arc2);
   
    //Add Score text to the gauge layer
    scoreText
    .transition().duration(1)
    .style("opacity", 1)
    .text('Score: '+(100*perc).toFixed(1)+'%');
  }


  
  var Needle = (function() {
    /** 
      * Helper function that returns the `d` value
      * for moving the needle
    **/
    var recalcPointerPos = function(perc) {
      var centerX, centerY, leftX, leftY, rightX, rightY, thetaRad, topX, topY;
      thetaRad = percToRad(perc / 2);
      centerX = 0;
      centerY = 0;
      topX = centerX - this.len * Math.cos(thetaRad);
      topY = centerY - this.len * Math.sin(thetaRad);
      leftX = centerX - this.radius * Math.cos(thetaRad - Math.PI / 2);
      leftY = centerY - this.radius * Math.sin(thetaRad - Math.PI / 2);
      rightX = centerX - this.radius * Math.cos(thetaRad + Math.PI / 2);
      rightY = centerY - this.radius * Math.sin(thetaRad + Math.PI / 2);
      return "M " + leftX + " " + leftY + " L " + topX + " " + topY + " L " + rightX + " " + rightY;
    };

    function Needle(cg) {
      this.cg = cg;
      this.len = width / 4;
      this.radius = this.len / 6;
    }

    Needle.prototype.render = function() {
      this.cg.append('circle').attr('class', 'needle-center').attr('cx', 0).attr('cy', 0).attr('r', this.radius);
      return this.cg.append('path').attr('class', 'needle').attr('d', recalcPointerPos.call(this, 0));
    };

    Needle.prototype.moveTo = function(perc) {
      var self,
          oldValue = this.perc || 0;

      this.perc = perc;
      self = this;

      // Reset pointer position
      this.cg.transition().delay(100).ease('quad').duration(200).select('.needle').tween('reset-progress', function() {
        return function(percentOfPercent) {
          var progress = (1 - percentOfPercent) * oldValue;
          
          repaintGauge(progress);
          return d3.select(this).attr('d', recalcPointerPos.call(self, progress));
        };
      });

      this.cg.transition().delay(300).ease('bounce').duration(1500).select('.needle').tween('progress', function() {
        return function(percentOfPercent) {
          var progress = percentOfPercent * perc;
          
          repaintGauge(progress);
          return d3.select(this).attr('d', recalcPointerPos.call(self, progress));
        };
      });

    };

    return Needle;

  })();

  //Create and render the needle.
  needle = new Needle(chart_cg);
  needle.render();
  //Move initial position to 0 upon load
  needle.moveTo(0.0);
  

  //Create a list of suggestions 
  sl = d3.select('.suggestions-list');
  width_sl = sl[0][0].offsetWidth; 
  height_sl = sl[0][0].offsetHeight;

  // Create SVG element
  svg_sl = sl.append('svg').attr('width', width_sl).attr('height', height_sl);
  
  // Add layer for the panel
  chart_sl = svg_sl.append('g');

  var Suggestions = (function(){
    function Suggestions(cg) {
      this.cg = cg;
    }

    Suggestions.prototype.showList = function(suggestions_list) {
      //Create custom linear d3 scale for determining height of text and bars
      var y = d3.scale.linear()
        .domain([0,10])
        .range([0,400]);

      //Append text based on if there are suggestions
      if(suggestions_list.length > 0){
         this.cg.append('text')
        .attr("x", 0) 
        .attr("y", 20)
        .style("font-size", "24px")
        .style('opacity', 0) 
        .style("fill", '#464A4F')
        .text("Tailored Suggestions For Your Headline: ")
        .transition()
        .ease("quad-out")
        .duration(500)
        .style('opacity', 1) 
        .delay(2000);
      } else {
        this.cg.append('text')
        .attr("x", 0) 
        .attr("y", 20)
        .style("font-size", "24px")
        .style('opacity', 0) 
        .style("fill", '#464A4F')
        .text("That's a pretty awesome headline! ")
        .transition()
        .ease("quad-out")
        .duration(500)
        .style('opacity', 1) 
        .delay(2000);
      }
       
      //Create the rectangles for holding suggestions  
      var bar = this.cg.selectAll(".bar")
      .data(suggestions_list)
      .enter().append("g")
      .attr("class", "bar"); 

      bar.append("rect")
      .attr("y", function(d,i) { return y(i)+30; })
      .attr("height", 0)
      .attr("width", 600)
      .style("fill","#1a9850")
      .transition()
        .ease("quad-out")
        .duration(500)
        .attr("height", 38)
        .delay(function(d,i) { 
          return (i+5)*500;
        });
        
      //Create the suggestion text  
      bar.append("text")
      .style("fill","#fff")
      .attr("x", 5)
      .attr("y", function(d,i) { return y(i)+55; })
      .text(function(d) { return d; });
    };
    return Suggestions;

  })();
  //Create the suggestion object from the panel
  suggestions = new Suggestions(chart_sl);

})();