---
title: "[new preprint] A fast and tiny current as common generator of slow regular pacemaking in brain and heart"
date: 2025-10-31
categories: [Research, Publications]
description: New collaborative work identifying a conserved pacemaker mechanism across neuronal and cardiac systems.
tags: [preprint, research, computational neuroscience, cardiac, pacemaking]
---

Excited to share our new collaborative preprint, now on [bioRxiv](https://doi.org/10.1101/2025.10.30.685563)!

## The discovery

We identified a **conserved pacemaker current** that drives slow, regular firing in both midbrain dopamine neurons and heart pacemaker cells. Using a pharmacological blocker (XG), we found this current is essential - blocking it completely silences pacemaking across species (rodent, human) and systems (brain, heart).

The current has remarkable properties:
- Fast-activating (nearly instantaneous)
- Small amplitude but crucial
- Voltage-dependent activation starting around -50 mV
- Conserved across neuronal and cardiac cells

## Modeling & validation

Our computational modeling (in Julia) showed that this small pacemaker conductance is sufficient for slow pacemaking, even in minimal models. Crucially, the voltage-dependence matters - replacing it with a linear leak cannot sustain stable slow pacemaking.

*Dynamic-clamp experiments validated the model*: injecting the modeled conductance into real neurons could silence, modulate, or rescue pacemaking as predicted.

## Why it matters

This is **the first demonstration of a shared pacemaker mechanism between brain and heart.** It challenges existing theories (Ca²⁺ clock, NALCN channels) and reveals a fundamental principle: the most important currents can be the smallest ones.

---

**Preprint**: [bioRxiv](https://doi.org/10.1101/2025.10.30.685563)

**Authors**: Arthur Fyon, Oleksandra Pavlova, Nick Schaar, Pietro Mesirca, Julien Brandoit, Sofian Ringlet, Alessio Franci, Matteo E. Mangoni, Jochen Roeper, Guillaume Drion, Vincent Seutin, Kevin Jehasse

**Contact**: [Kevin.Jehasse@uliege.be](mailto:Kevin.Jehasse@uliege.be), [afyon@uliege.be](mailto:afyon@uliege.be)