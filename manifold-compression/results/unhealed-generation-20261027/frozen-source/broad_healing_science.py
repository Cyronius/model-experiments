"""Original short explanations, checked against the linked primary sources.

One target per topic; resampling does not create additional topic diversity.
None covers the frozen photosynthesis or heat-conduction evaluation prompts.
"""
SCIENCE = [
('seasons', 'Why do the Northern and Southern Hemispheres have opposite seasons? Explain in two short paragraphs.',
 "Earth's rotation axis is tilted relative to its orbit around the Sun. As Earth travels around that orbit, one hemisphere receives sunlight more directly and has longer daylight hours. That hemisphere experiences summer while the other receives less direct sunlight and has shorter days.\n\nAbout half a year later, the arrangement reverses. The hemispheres therefore have opposite seasons. Changes in Earth's distance from the Sun are not the main cause: both hemispheres share essentially the same distance at a given time, yet their seasons differ.",
 'https://spaceplace.nasa.gov/seasons/en/'),
('moon', 'Why does the Moon appear to change shape? Distinguish ordinary phases from an eclipse.',
 "The Moon reflects sunlight. The Sun illuminates roughly half of the Moon at any moment, but the fraction of that illuminated half visible from Earth changes as the Moon orbits us. We call these changing views the lunar phases. The Moon itself is not changing shape.\n\nAt new moon, the illuminated side mostly faces away from Earth. At full moon, it faces toward us. Ordinary phases are not caused by Earth's shadow. A lunar eclipse is a separate event in which the Moon passes through that shadow.",
 'https://science.nasa.gov/moon/moon-phases/'),
('weather', 'How can a cold week occur while the climate is warming? Explain the distinction clearly.',
 "Weather describes atmospheric conditions over short periods in a particular place: a cold morning, a storm, or a rainy week. Climate describes longer-term patterns, including typical conditions and their variability over many years. They describe different time scales.\n\nA warming climate does not eliminate day-to-day or year-to-year fluctuations. A particular place can still have a cold week even when long-term average temperatures are rising. To assess a climate trend, scientists examine many observations over extended periods rather than treating one local weather event as decisive.",
 'https://science.nasa.gov/climate-change/faq/whats-the-difference-between-weather-and-climate/'),
('earthquake', 'Explain how movement along a fault can cause an earthquake.',
 "Tectonic plates move, but friction can prevent the rocks on opposite sides of a fault from sliding smoothly. Stress builds as surrounding rocks deform while part of the fault remains stuck.\n\nWhen that stress overcomes the resistance, the fault can slip suddenly. Released energy travels through Earth as seismic waves, producing the shaking felt during an earthquake. The earthquake is the sudden slip and resulting release of energy, not simply the existence of a crack in the ground.",
 'https://www.usgs.gov/faqs/what-earthquake-and-what-causes-them-happen'),
('magma', 'What is the difference between magma and lava?',
 "Magma is molten rock beneath Earth's surface. When molten rock emerges onto the surface, it is called lava. The distinction refers to its location.\n\nFor example, molten material stored below a volcano is magma. Material flowing from an eruption is lava. Both can eventually cool and solidify into rock; calling a flow lava does not mean it is a completely different substance from the magma that supplied it.",
 'https://www.usgs.gov/faqs/what-difference-between-magma-and-lava'),
('igneous', 'Why do some igneous rocks have large crystals while others have tiny crystals?',
 "Igneous rocks form as molten material cools and solidifies. Cooling slowly underground gives crystals more time to grow, so intrusive rocks can develop relatively large, visible crystals. Granite is a familiar example.\n\nLava cooling rapidly at the surface gives crystals much less time to grow. Extrusive rocks therefore often contain very small crystals; sufficiently rapid cooling can produce a glassy texture. The contrast reflects cooling conditions, rather than a rule that every rock contains equally large crystals.",
 'https://www.usgs.gov/faqs/what-are-igneous-rocks'),
('eclipse', 'Compare solar and lunar eclipses and explain why they do not occur every month.',
 "In a solar eclipse, the Moon passes between Earth and the Sun and blocks the Sun from view for observers in the Moon's shadow. In a lunar eclipse, Earth is between the Sun and Moon, and the Moon passes through Earth's shadow. A solar eclipse requires a new moon; a lunar eclipse requires a full moon.\n\nThese alignments do not happen at every new or full moon because the Moon's orbit is tilted relative to Earth's orbit around the Sun. Usually the bodies do not line up closely enough for their shadows to produce an eclipse.",
 'https://science.nasa.gov/moon/eclipses/'),
('meteor', 'Explain the difference between a meteoroid, a meteor, and a meteorite.',
 "A meteoroid is a small rocky or metallic object traveling in space. If it enters Earth's atmosphere and produces a bright streak, that visible event is called a meteor. The popular phrase shooting star describes this appearance; it is not a star falling from the sky.\n\nIf some of the incoming material survives its passage through the atmosphere and reaches the ground, the surviving object is a meteorite. The three terms distinguish the object in space, the atmospheric phenomenon, and material that reaches the surface.",
 'https://science.nasa.gov/solar-system/meteors-meteorites/'),
]
