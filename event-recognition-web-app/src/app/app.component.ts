import { Component, OnInit, AfterViewInit, Pipe } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import * as Chart from 'chart.js';
import { DomSanitizer } from '@angular/platform-browser';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements AfterViewInit {
  // *********************
  // ** Public Members ***
  // *********************
  public title = 'Event Recognition Web App';

  // *********************
  // ** Private Members **
  // *********************
  private m_selectedFile: File;
  private m_resultsLoading = false;
  private m_chart: Chart;
  private m_config = {
    type: 'horizontalBar',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Probability',
          backgroundColor: ['#3e95cd', '#8e5ea2', '#3cba9f', '#e8c3b9', '#c45850'],
          data: []
        }
      ]
    },
    options: {
      legend: { display: false },
      title: {
        display: true,
        text: 'Top-5 classes predicted'
      },
      scales: {
        xAxes: [{
            display: true,
            ticks: {
                suggestedMin: 0,
                suggestedMax: 1
            }
        }]
      },
      responsive: true,
      maintainAspectRatio: false
    }
  };

  // *************************************
  // ** Construction and Initialization **
  // *************************************
  constructor(private m_http: HttpClient,
    private sanitizer: DomSanitizer) {
  }
  ngAfterViewInit() {
    this.m_chart = new Chart(document.getElementById('bar-chart-horizontal'), this.m_config);
  }

  // *********************
  // ** Public Methods ***
  // *********************
  public get resultsLoading(): boolean {
    return this.m_resultsLoading;
  }
  public get isImageSelected(): boolean {
    return !!this.m_selectedFile;
  }
  public get imageSrc(): any {
    if (!!this.m_selectedFile) {
      const url = URL.createObjectURL(this.m_selectedFile);
      return this.sanitizer.bypassSecurityTrustUrl(url);
    }
  }

  // *********************
  // ** Private Methods **
  // *********************
  onFileChanged(event) {
    this.m_selectedFile = event.target.files[0];
  }
  onUpload() {
    this.m_resultsLoading = true;
    const reader = new FileReader();
    const formData = new FormData();
    formData.append('image', this.m_selectedFile);
    this.m_http.post('http://localhost:5000/predict', formData).subscribe(
      (res: any) => {
        if (res.predictions && res.predictions.length === 5) {
          const labels = [];
          for (let i = 0; i < 5; i++) {
            labels.push(res.predictions[i].label);
          }
          this.m_config.data.labels = labels;
          const probabilities = [];
          for (let i = 0; i < 5; i++) {
            probabilities.push(res.predictions[i].probability);
          }
          this.m_config.data.datasets[0].data = probabilities;
          this.m_resultsLoading = false;
          this.m_chart.update();
        }
      },
      err => {
        console.log('Error occured');
      }
    );
  }
}
