<script lang="ts">
  import { architectureContent, diagrams } from './aboutContent';
</script>

<section class="section">
  <h2 class="section-title">{architectureContent.title}</h2>

  <h3 class="section-subtitle">{architectureContent.systemOverview.title}</h3>
  <p class="section-desc">{architectureContent.systemOverview.description}</p>
  <div class="diagram">
    <pre class="ascii-art">{diagrams.systemArchitecture}</pre>
  </div>

  <h3 class="section-subtitle">{architectureContent.agentFlow.title}</h3>
  <p class="section-desc">{architectureContent.agentFlow.description}</p>
  <div class="diagram">
    <pre class="ascii-art">{diagrams.reactFlow}</pre>
  </div>
</section>
