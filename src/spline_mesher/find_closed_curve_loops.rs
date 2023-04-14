fn find_closed_curve_loops(
  lines_lower_dict: HashMap<i32, (i32, i32)>,
  lines_upper_dict: HashMap<i32, (i32, i32)>,
  lines_intersurf_dict: HashMap<i32, (i32, i32)>,
) -> Vec<Vec<i32>> {
  let mut closed_curve_loops = Vec::new();
  for (l1, l3) in itertools::iproduct(lines_lower_dict.keys(), lines_upper_dict.keys()) {
      // For each pair of lines from the upper and lower surfaces
      let (l1_start, l1_end) = lines_lower_dict.get(l1).unwrap();
      let (l3_start, l3_end) = lines_upper_dict.get(l3).unwrap();
      for (l4, l2) in itertools::iproduct(lines_intersurf_dict.keys(), lines_intersurf_dict.keys()) {
          // For each pair of lines from the intersection surface
          if l4 == l2 {
              // Skip if the lines are the same
              continue;
          }
          let (l2_start, l2_end) = lines_intersurf_dict.get(l2).unwrap();
          let (l4_start, l4_end) = lines_intersurf_dict.get(l4).unwrap();
          if (l1_start == l2_end || l1_end == l2_start) && (l1_start == l4_start || l1_end == l4_end)
          {
              // If the start or end vertex of one line from the upper/lower surface is the start or
              // end vertex of one line from the intersection surface, and the start or end vertex
              // of the other line from the upper/lower surface is the start or end vertex of the
              // other line from the intersection surface, then we have a closed curve loop
              if (l3_start == l2_start || l3_end == l2_end) && (l3_start == l4_end || l3_end == l4_start)
              {
                  // If the start or end vertex of one line from the upper/lower surface is the start or
                  // end vertex of one line from the intersection surface, and the start or end vertex
                  // of the other line from the upper/lower surface is the start or end vertex of the
                  // other line from the intersection surface, then we have a closed curve loop
                  let cl_s = [*l1, *l2, *l3, *l4];
                  closed_curve_loops.push(cl_s);
              }
          }
      }
  }
  closed_curve_loops
}
