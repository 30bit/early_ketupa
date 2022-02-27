use {
    glam::{vec2, Mat2, Vec2},
    lyon_tessellation::{
        path::traits::PathBuilder, BuffersBuilder, FillOptions, FillTessellator, FillVertex,
        VertexBuffers,
    },
    macroquad::{
        camera::{set_camera, Camera2D},
        color::{Color, BLANK, BLUE, DARKGRAY, GRAY, GREEN, LIGHTGRAY, RED, WHITE},
        input::{
            is_mouse_button_down, is_mouse_button_pressed, is_mouse_button_released,
            mouse_position, MouseButton,
        },
        models::{draw_mesh, Mesh, Vertex},
        shapes::{draw_circle, draw_line},
        ui::{root_ui, Skin},
        window::{clear_background, next_frame, screen_height, screen_width},
    },
    std::iter::once,
};

#[macroquad::main("Owlet Demo")]
async fn main() {
    skin_ui();

    let mut selection = None;
    let mut points = default_points();
    let mut look_degrees = 90;
    let mut rotation_sign = 0;

    loop {
        clear_background(LIGHTGRAY);

        process_ui(&mut points, &mut look_degrees, &mut rotation_sign);

        let mouse = camera_mouse();
        let point_near_mouse = point_near(&points, mouse);
        let line_near_mouse = point_near_mouse.or_else(|| line_near(&points, mouse));

        let dir = owlet::dirs::degrees(look_degrees as f32);
        let inv = Mat2::from_angle((look_degrees as f32 - 90.0).to_radians());
        let look = owlet::Look::new(|i: usize| points[i].into(), dir);

        let minmax = owlet::core::Minmax::new(look, 0..points.len());
        let ((min_index, min_norm), (max_index, max_norm)) = minmax.unwrap_pair();

        let mut marker_color = BLANK;
        let view = {
            let mut prev_index = max_index;
            let iter = owlet::core::Iter::new(look, 0..points.len(), minmax).inspect(
                |(category, segment)| {
                    process_segment(
                        &points,
                        &mut prev_index,
                        &mut marker_color,
                        point_near_mouse,
                        line_near_mouse,
                        category,
                        segment,
                    )
                },
            );
            owlet::View::with_iter(look, 0..points.len(), iter)
        };

        let shadow = {
            let far = |norm| inv * vec2(norm, 2.0);
            shadow_points(&points, min_index, max_index)
                .chain([min_norm, max_norm].into_iter().map(far))
        };

        let light = {
            let far = |norm| inv * vec2(norm, -2.0);
            light_points(look, points.len(), &view.hull)
                .chain([max_norm, min_norm].into_iter().map(far))
        };

        draw_polygon(light, WHITE);
        draw_polygon(shadow, GRAY);
        draw_polygon(points.iter().cloned(), DARKGRAY);

        draw_mouse(
            &points,
            point_near_mouse,
            line_near_mouse,
            mouse,
            marker_color,
        );

        process_mouse(
            &mut points,
            &mut selection,
            point_near_mouse,
            line_near_mouse,
            mouse,
        );

        next_frame().await
    }
}

fn process_ui(points: &mut Vec<Vec2>, look_degrees: &mut i32, rotation_sign: &mut i32) {
    let label = |text: &str| text.lines().for_each(|s| root_ui().label(None, s));

    label(&format!(
        "look direction is {}",
        if *rotation_sign == 0 {
            (0..360)
                .step_by(90)
                .zip(["right", "up", "left", "down"])
                .find(|(degrees, _)| *degrees == *look_degrees as u32)
                .map(|(_, dir)| dir)
        } else {
            None
        }
        .unwrap_or(&format!("{} degrees", look_degrees))
    ));

    fn button(text: impl AsRef<str>, mut f: impl FnMut()) {
        if root_ui().button(None, text.as_ref()) {
            f()
        }
    }

    let mut alter_look = |d| *look_degrees = (*look_degrees + 360 + d) % 360;

    button("rotate once counterclockwise", || alter_look(15));
    button("rotate once clockwise", || alter_look(-15));

    if *rotation_sign == 0 {
        button("start clockwise rotation", || *rotation_sign = -1);
        button("start counterclockwise rotation", || *rotation_sign = 1);
    } else {
        alter_look(*rotation_sign);
        button("reverse rotation direction", || *rotation_sign *= -1);
        button("end rotation", || *rotation_sign = 0);
    }

    button("reset points", || *points = default_points());

    label(" \nextend  with left drag\ninsert with left click\nremove with right click\n \n");
    if is_clockwise(points) {
        label("polygon became clockwise\ncounterclockwise is expected\nconsider resetting points");
    }
}

fn process_segment(
    points: &[Vec2],
    prev_index: &mut usize,
    marker_color: &mut Color,
    point_near_mouse: Option<usize>,
    line_near_mouse: Option<usize>,
    category: &owlet::core::Category,
    segment: &owlet::core::Segment<usize, f32>,
) {
    let within = |start, end| {
        (start..end).contains(&point_near_mouse.or(line_near_mouse).unwrap_or(usize::MAX))
    };

    if (*prev_index < segment.to.0) ^ (within(*prev_index, points.len()) | within(0, segment.to.0))
        | within(*prev_index, segment.to.0)
    {
        *marker_color = match category {
            owlet::core::Category::Obscuring => RED,
            owlet::core::Category::Contiguous => GREEN,
            owlet::core::Category::Partial => BLUE,
        }
    }
    *prev_index = segment.to.0;
}

fn process_mouse(
    points: &mut Vec<Vec2>,
    selection: &mut Option<usize>,
    point_near: Option<usize>,
    line_near: Option<usize>,
    mouse: Vec2,
) {
    if is_mouse_button_down(MouseButton::Left) {
        if let Some(index) = *selection {
            points[index] = mouse;
        } else if let Some(index) = point_near {
            *selection = Some(index);
        } else if is_mouse_button_pressed(MouseButton::Left) {
            if let Some(index) = line_near {
                points.insert(index + 1, mouse)
            }
        }
    } else if is_mouse_button_released(MouseButton::Left) {
        *selection = None;
    } else if is_mouse_button_pressed(MouseButton::Right) && points.len() > 3 {
        if let Some(index) = point_near {
            points.remove(index);
        }
    }
}

fn light_points<'a>(
    look: impl owlet::core::Norm<usize, f32> + owlet::core::Point<usize, f32> + Copy + 'a,
    len: usize,
    hull: &'a owlet::core::Hull<usize, f32>,
) -> impl Iterator<Item = Vec2> + 'a {
    let decrement = |i: usize| i.checked_sub(1).unwrap_or(len - 1);
    let with_norm = |i| (i, look.norm(i));

    let extrusion = |start, end| {
        hull.indices[start..end]
            .iter()
            .zip(hull.indices[start..end].iter().skip(1))
            .map(|(&prev, &next)| {
                if decrement(next) == prev {
                    (prev, None)
                } else {
                    let successor = (prev + 1) % len;
                    let s = owlet::core::Segment::new(with_norm(prev), with_norm(successor));
                    let mid = s.lerp(look, look.norm(next));
                    (prev, Some(mid))
                }
            })
            .chain(once((hull.indices[end - 1], None)))
    };

    let extrude = |light: &mut Vec<_>, extrusion| {
        for (i, mid) in extrusion {
            light.push(look.point(i));
            if let Some(mid) = mid {
                light.push(mid);
            }
        }
    };

    let mut light = Vec::with_capacity(len);

    let main = once(&(0, 0))
        .chain(hull.partials.iter())
        .zip(hull.partials.iter())
        .map(|(prev, next)| (next.0, prev.1, next.1))
        .map(|(i, start, end)| (i, end, extrusion(start, end)));

    for (i, j, extrusion) in main {
        extrude(&mut light, extrusion);
        let partial = owlet::core::Segment::new(with_norm(decrement(i)), with_norm(i));
        light.push(partial.lerp(look, hull.norms[j - 1]));
        light.push(partial.lerp(look, hull.norms[j]));
    }

    let remainder = extrusion(
        hull.partials.last().map(|(_, j)| *j).unwrap_or(0),
        hull.indices.len(),
    );

    extrude(&mut light, remainder);

    light.dedup();
    light.into_iter().map(Into::into)
}

fn shadow_points(
    points: &'_ [Vec2],
    min_index: usize,
    max_index: usize,
) -> impl Iterator<Item = Vec2> + '_ {
    if min_index < max_index {
        points[max_index..].iter().chain(&points[..=min_index])
    } else {
        points[max_index..=min_index].iter().chain(&[])
    }
    .cloned()
}

fn default_points() -> Vec<Vec2> {
    const H: f32 = 0.6;
    const Y: f32 = H / 3.0;
    const X: f32 = H / 1.73205;
    vec![vec2(-X, H), vec2(0.0, -Y), vec2(X, H), vec2(0.0, Y)]
}

fn draw_mouse(
    points: &[Vec2],
    point_near_mouse: Option<usize>,
    line_near_mouse: Option<usize>,
    mouse: Vec2,
    color: Color,
) {
    let tangent = |from: Vec2, to: Vec2, color| draw_line(from.x, from.y, to.x, to.y, 0.01, color);
    let vertex = |center: Vec2, color| draw_circle(center.x, center.y, 0.007, color);
    if let Some(index) = point_near_mouse {
        vertex(points[index], color)
    } else if let Some(index) = line_near_mouse {
        let from = points[index];
        let to = points[(index + 1) % points.len()];
        let l = (to - from).normalize();
        let center = from + l.dot(mouse - from) * l;
        let dir = (center - to).normalize();
        let r = dir * center.distance(to).min(center.distance(from)).min(0.04);
        tangent(center - r, center + r, color)
    }
}

fn draw_polygon(iter: impl IntoIterator<Item = Vec2>, color: Color) {
    let mut iter = iter.into_iter();
    let hint = iter.size_hint();
    let cap = hint.1.unwrap_or(hint.0);
    let first = iter.next().unwrap();
    let mut vb = VertexBuffers::with_capacity(cap, cap.saturating_sub(2) * 3);
    let mut ft = FillTessellator::new();
    let fo = FillOptions::default().with_tolerance(0.00001);

    fn vertex(p: impl Into<mint::Point2<f32>>, color: Color) -> Vertex {
        Vertex {
            position: Vec2::from(p.into()).extend(0.0),
            uv: Vec2::ZERO,
            color,
        }
    }

    let mut bb = BuffersBuilder::new(&mut vb, |p: FillVertex<'_>| vertex(p.position(), color));
    let mut fb = ft.builder(&fo, &mut bb);

    fn point<P: From<mint::Point2<f32>>>(p: impl Into<mint::Point2<f32>>) -> P {
        p.into().into()
    }

    fb.begin(point(first));
    for p in iter.map(point) {
        fb.line_to(p);
    }
    fb.line_to(point(first));
    fb.end(true);
    fb.build().unwrap();

    draw_mesh(&Mesh {
        vertices: vb.vertices,
        indices: vb.indices,
        texture: None,
    });
}

fn camera_mouse() -> Vec2 {
    let camera = Camera2D {
        zoom: vec2(screen_height() / screen_width(), 1.0),
        ..Default::default()
    };
    set_camera(&camera);
    camera.screen_to_world(mouse_position().into())
}

fn point_near(points: &[Vec2], target: Vec2) -> Option<usize> {
    points.iter().position(|&o| target.distance(o) <= 0.02)
}

fn line_near(points: &[Vec2], target: Vec2) -> Option<usize> {
    points
        .iter()
        .zip(points.iter().skip(1))
        .enumerate()
        .chain({
            let last = points.len() - 1;
            once((last, (&points[last], &points[0])))
        })
        .filter_map(|(index, (&from, &to))| {
            let from_to = to - from;
            let length_recip = from_to.length_recip();
            if length_recip < 16.0 {
                Some((index, from, to, from_to * length_recip))
            } else {
                None
            }
        })
        .filter_map(|(index, from, to, from_to)| {
            let l = target - from;
            let d = l.dot(from_to);
            if d.is_sign_positive() != (target - to).dot(from_to).is_sign_positive() {
                let h = l - from_to * d;
                Some((index, h.length()))
            } else {
                None
            }
        })
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .and_then(
            |(index, distance)| {
                if distance <= 0.01 {
                    Some(index)
                } else {
                    None
                }
            },
        )
}

fn is_clockwise(points: &[Vec2]) -> bool {
    points
        .iter()
        .zip(points.iter().skip(1))
        .chain(once((points.last().unwrap(), points.first().unwrap())))
        .map(|(from, to)| (to.x - from.x) * (from.y + to.y))
        .sum::<f32>()
        .is_sign_positive()
}

fn skin_ui() {
    let button_style = root_ui()
        .style_builder()
        .color(BLANK)
        .color_hovered(WHITE)
        .color_clicked(WHITE)
        .build();
    let label_style = root_ui().style_builder().build();
    let skin = Skin {
        button_style,
        label_style,
        ..root_ui().default_skin()
    };
    root_ui().push_skin(&skin);
}
